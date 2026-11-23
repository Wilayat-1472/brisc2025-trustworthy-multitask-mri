from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from brisc_mri.data import preprocess, load_manifest, development_splits, BRISCMultiTaskDataset
from brisc_mri.inference import Predictor, conformal_threshold, prediction_set
from brisc_mri.model import PlaneConditionedMultiTaskUNet
from brisc_mri.visualization import render_result

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def test_real_manifest_group_integrity():
    df = load_manifest(ROOT / 'data/brisc2025_clean.csv', check_files=False)
    parts = development_splits(df)
    assert sum(map(len, parts)) == 6000
    groups = [set(p.pseudo_patient_id) for p in parts]
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert a.isdisjoint(b)


def test_dataset_rebases_paths_and_matches_preprocessing(tmp_path):
    folder = tmp_path / 'classification_task/train/glioma'
    folder.mkdir(parents=True)
    img = Image.new('L', (32, 32), 150)
    img.save(folder / 'scan.png')
    mask = tmp_path / 'segmentation_task/train/masks/mask.png'
    mask.parent.mkdir(parents=True)
    Image.new('L', (32, 32), 255).save(mask)
    df = pd.DataFrame([{'image_path': '/kaggle/input/brisc2025/brisc2025/classification_task/train/glioma/scan.png',
                        'mask_path': '/old/segmentation_task/train/masks/mask.png', 'tumor_label': 'glioma',
                        'plane_code': 'ax', 'pseudo_patient_id': 1, 'strict_protocol': 'train'}])
    manifest = tmp_path / 'manifest.csv'
    df.to_csv(manifest, index=False)
    ds = BRISCMultiTaskDataset(load_manifest(manifest, tmp_path))
    x, y, label, plane = ds[0]
    assert torch.equal(x, preprocess(img))
    assert y.shape == (1, 256, 256) and y.min() == 1
    assert label == 0 and plane == 0
    df.loc[0, 'mask_path'] = ''
    df.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match='require ground-truth masks'):
        load_manifest(manifest, tmp_path)


def test_conformal_finite_sample_rank_and_empty_sets():
    probs = np.array([[0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0], [0.7, 0.3, 0, 0]])
    assert conformal_threshold(probs, [0, 0, 0], alpha=0.5) == pytest.approx(0.2)
    assert conformal_threshold(probs, [0, 0, 0], alpha=0.05) == 1
    assert prediction_set([0.4, 0.3, 0.2, 0.1], 0.05) == []
    assert prediction_set([0.4, 0.3, 0.2, 0.1], None) is None


def test_model_checkpoint_and_full_inference(tmp_path):
    # Synthetic weights verify execution only; they are never distributed as trained weights.
    torch.manual_seed(7)
    model = PlaneConditionedMultiTaskUNet().eval()
    with torch.no_grad():
        model.cls_head[-1].weight.zero_()
        model.cls_head[-1].bias.copy_(torch.tensor([0., 0., 4., 0.]))
        cls, seg = model(torch.zeros(1, 3, 64, 64), torch.tensor([0]))
    assert cls.shape == (1, 4) and seg.shape == (1, 1, 64, 64)
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    predictor = Predictor(path)
    image = Image.new('L', (80, 80), 120)
    result = predictor.predict(image, 'ax', 2)
    assert result['predicted_class'] == result['cam_target'] == 'pituitary'
    assert result['prediction_set'] is None
    assert result['mask'].shape == result['uncertainty'].shape == result['cam'].shape == (256, 256)
    assert np.isfinite(result['cam']).all()
    assert result['probabilities'].sum() == pytest.approx(1)
    assert not predictor.model.enc4._forward_hooks
    assert not any(m.training for m in predictor.model.modules())
    assert render_result(image, result).startswith(b'\x89PNG')
    torch.save({'model_state_dict': model.state_dict(), 'q_hat': 0.0, 'alpha': 0.05}, path)
    calibrated = Predictor(path).predict(image, n_passes=1)
    assert calibrated['prediction_set'] == [] and calibrated['review_required'] is True
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path / 'missing.pt')


def test_interface_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("BRISC_CHECKPOINT", str(tmp_path / "missing.pt"))
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert app.button[0].disabled
    assert any('Model weights are needed' in x.value for x in app.info)
    app.sidebar.selectbox[0].select('Axial').run()
    assert not app.exception
    assert app.button[0].disabled


def test_notebooks_are_output_free_and_code_compiles():
    for path in (ROOT / 'notebooks').glob('*.ipynb'):
        notebook = json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                assert not cell.get('outputs')
                source = ''.join(cell['source'])
                if not any(line.startswith(('!', '%')) for line in source.splitlines()):
                    compile(source, str(path), 'exec')


def test_interface_prediction_and_stale_result_clear(tmp_path, monkeypatch):
    from io import BytesIO
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    path = tmp_path / 'weights.pt'
    torch.save(PlaneConditionedMultiTaskUNet().state_dict(), path)
    image = BytesIO()
    Image.new('L', (48, 48), 130).save(image, format='PNG')
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: image)
    monkeypatch.setenv('BRISC_CHECKPOINT', str(path))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.sidebar.selectbox[0].select('Axial')
    app.sidebar.slider[0].set_value(1)
    app.run()
    assert not app.button[0].disabled
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.metric) == 2
    assert len(app.get('download_button')) == 3
    assert any('calibration is unavailable' in i.value for i in app.info)
    app.sidebar.selectbox[0].select('Coronal').run()
    assert not app.exception and not app.metric
