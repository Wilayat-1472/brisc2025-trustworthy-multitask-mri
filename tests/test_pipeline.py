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


@pytest.mark.parametrize('probs,labels', [
    ([[0.9, 0.1, 0, 0]], [-1]),
    ([[0.9, 0.1, 0, 0]], [0.5]),
    ([[float('nan'), 0.1, 0, 0]], [0]),
    ([[0.9, 0.2, 0, 0]], [0]),
    ([[0.9, 0.1]], [0]),
    ([[0.9, 0.1, 0, 0]], [0, 1]),
])
def test_calibration_rejects_invalid_probabilities_and_labels(probs, labels):
    with pytest.raises(ValueError):
        conformal_threshold(probs, labels)


def test_manifest_rejects_duplicate_paths_and_cross_split_hashes(tmp_path):
    rows = [dict(image_path=f'{i}.png', mask_path='', tumor_label='no_tumor',
                 plane_code='ax', pseudo_patient_id=i, strict_protocol=split,
                 sha256=str(i)) for i, split in enumerate(['train', 'val_cal', 'test'])]
    path = tmp_path / 'manifest.csv'
    frame = pd.DataFrame(rows)
    frame.loc[2, 'image_path'] = './0.png'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='duplicate image paths'):
        load_manifest(path, check_files=False)
    frame.loc[2, 'image_path'] = '2.png'
    frame.loc[2, 'sha256'] = '0'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='hashes overlap'):
        load_manifest(path, check_files=False)


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
    assert len(app.metric) == 4
    assert len(app.get('download_button')) == 3
    assert any('calibration is unavailable' in i.value for i in app.info)
    app.sidebar.selectbox[0].select('Coronal').run()
    assert not app.exception and not app.metric


def test_src_model_and_predict_joblib_bundle(tmp_path):
    import joblib
    from src.model import PlaneConditionedMultiTaskUNet as SrcUNet
    from predict import load_brisc_bundle, predict_slice

    model = SrcUNet(num_classes=4, num_planes=3, pretrained=False).eval()
    bundle = {
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'classes': ['glioma', 'meningioma', 'pituitary', 'no_tumor'],
        'planes': ['axial', 'coronal', 'sagittal'],
        'img_size': 256,
        'conformal_qhat': 0.0579,
        'norm_mean': [0.485, 0.456, 0.406],
        'norm_std': [0.229, 0.224, 0.225],
    }
    bundle_path = tmp_path / 'test_bundle.joblib'
    joblib.dump(bundle, bundle_path, compress=1)

    loaded_model, loaded_bundle = load_brisc_bundle(bundle_path)
    assert loaded_bundle['classes'] == ['glioma', 'meningioma', 'pituitary', 'no_tumor']

    sample_img = tmp_path / 'sample.png'
    Image.new('L', (64, 64), 140).save(sample_img)
    out_vis = tmp_path / 'overlay.png'
    res = predict_slice(sample_img, plane='axial', out_vis=str(out_vis), bundle_path=bundle_path)
    assert res['pred_class'] in bundle['classes']
    assert out_vis.is_file() and out_vis.stat().st_size > 0


def test_interface_with_joblib_bundle(tmp_path, monkeypatch):
    from io import BytesIO
    import joblib
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from src.model import PlaneConditionedMultiTaskUNet as SrcUNet

    model = SrcUNet(num_classes=4, num_planes=3, pretrained=False).eval()
    bundle = {
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'classes': ['glioma', 'meningioma', 'pituitary', 'no_tumor'],
        'planes': ['axial', 'coronal', 'sagittal'],
        'img_size': 256,
        'conformal_qhat': 0.0579,
        'norm_mean': [0.485, 0.456, 0.406],
        'norm_std': [0.229, 0.224, 0.225],
    }
    bundle_path = tmp_path / 'test_bundle.joblib'
    joblib.dump(bundle, bundle_path, compress=1)

    image = BytesIO()
    Image.new('L', (48, 48), 130).save(image, format='PNG')
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: image)
    monkeypatch.setenv('BRISC_CHECKPOINT', str(bundle_path))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.sidebar.slider[0].set_value(1)
    app.run()
    assert not app.button[0].disabled
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.metric) == 4
    assert len(app.get('download_button')) == 3
