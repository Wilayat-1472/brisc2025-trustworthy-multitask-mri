"""Run with: streamlit run app.py"""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import os
import pickle

import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch

from brisc_mri.data import CLASS_NAMES
from brisc_mri.inference import Predictor
from brisc_mri.visualization import render_result

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='BRISC MRI · Research workspace', page_icon='🧠', layout='wide')
st.markdown('''<style>
.stApp {background: #0b1220; color: #e5edf8;}
[data-testid="stSidebar"] {background: #111c2e;}
h1 {letter-spacing: -0.04em;} .stButton button {border-radius: 8px;}
</style>''', unsafe_allow_html=True)
st.caption('BRISC 2025 / MULTI-TASK MRI')
st.title('Brain MRI research workspace')
st.write('Explore tumor classification, segmentation and model uncertainty from a single T1 MRI slice.')
st.caption('Research demonstration. Outputs are model estimates and are not validated for clinical diagnosis.')

with st.sidebar:
    st.header('Model & scan settings')
    checkpoint = st.text_input('Checkpoint path', os.environ.get('BRISC_CHECKPOINT', str(ROOT / 'checkpoints/best_brisc_model.pt')))
    plane_name = st.selectbox('Anatomical plane', ['Axial', 'Coronal', 'Sagittal'], index=None, placeholder='Select the scan orientation')
    n_passes = st.slider('MC dropout passes', 1, 30, 10)
    st.caption('More passes provide a steadier uncertainty estimate and take longer.')
    st.divider()
    st.markdown('**Expected input**\n\nOne PNG or JPEG T1 MRI slice. DICOM and NIfTI volumes must be converted first.')
    st.caption('Uploads are processed in memory and are not written to the project.')

scan_tab, about_tab = st.tabs(['Analyze a slice', 'Project & setup'])
with about_tab:
    st.subheader('Plane-conditioned multi-task U-Net')
    st.write('A shared ResNet34 encoder feeds a four-class classifier and a segmentation decoder. FiLM conditions the bottleneck on anatomical plane.')
    st.code('python -m pip install -e ".[dev]"\npython -m brisc_mri.prepare\nstreamlit run app.py', language='bash')
    st.write('Copy your trained best_brisc_model.pt from Colab into checkpoints/. Original notebook state dictionaries are supported. Conformal sets are available only when calibration metadata accompanies the weights.')
    st.image(str(ROOT / 'assets/brisc2025_clinical_diagnostics.png'), caption='Historical notebook visualization; this is not an uploaded-scan prediction.')

with scan_tab:
    ready = Path(checkpoint).expanduser().is_file()
    if not ready:
        st.info('Model weights are needed to analyze a scan. Copy best_brisc_model.pt from Colab to checkpoints/, or set its path in the sidebar.')
    uploaded = st.file_uploader('Upload an MRI slice', type=['png', 'jpg', 'jpeg'])
    image = None
    content = None
    if uploaded is not None:
        try:
            content = uploaded.getvalue()
            if len(content) > 20 * 1024 * 1024:
                raise ValueError('Please use an image smaller than 20 MB.')
            with Image.open(BytesIO(content)) as source:
                if source.width * source.height > 20_000_000:
                    raise ValueError('Please use an image with fewer than 20 million pixels.')
                image = source.convert('L')
            st.image(image, caption='Input MRI slice', width=320)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
            st.error(f'Could not open this image: {error}')
    path = Path(checkpoint).expanduser()
    signature = (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size) if ready else None
    input_key = (hashlib.sha256(content).hexdigest() if content else None, signature, plane_name, n_passes)
    if st.session_state.get('input_key') != input_key:
        st.session_state.pop('analysis', None)
    st.session_state['input_key'] = input_key
    if st.button('Analyze MRI slice', type='primary', disabled=not (ready and image is not None and plane_name)):
        try:
            # Keep one model per session; MC dropout and hooks mutate module state.
            if st.session_state.get('model_signature') != signature:
                st.session_state['predictor'] = Predictor(path, 'cuda' if torch.cuda.is_available() else 'cpu')
                st.session_state['model_signature'] = signature
            predictor = st.session_state['predictor']
            with st.spinner('Computing classification, segmentation, uncertainty and Grad-CAM…'):
                result = predictor.predict(image, {'Axial': 'ax', 'Coronal': 'co', 'Sagittal': 'sa'}[plane_name], n_passes)
                figure = render_result(image, result)
            report = {'predicted_class': result['predicted_class'], 'class_probabilities': dict(zip(CLASS_NAMES, map(float, result['probabilities']))),
                      'prediction_set': result['prediction_set'], 'review_required': result['review_required'],
                      'q_hat': predictor.q_hat, 'alpha': predictor.alpha,
                      'plane': plane_name, 'mc_passes': n_passes, 'research_only': True}
            mask_buffer = BytesIO()
            Image.fromarray((result['mask'] > 0.5).astype(np.uint8) * 255).save(mask_buffer, format='PNG')
            st.session_state['analysis'] = (report, figure, mask_buffer.getvalue())
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError) as error:
            st.error(f'Analysis failed: {error}')
    if 'analysis' in st.session_state:
        report, figure, mask = st.session_state['analysis']
        left, right = st.columns([1, 2])
        left.metric('Predicted class', report['predicted_class'].replace('_', ' ').title())
        left.metric('Top class probability', f"{max(report['class_probabilities'].values()):.1%}")
        right.bar_chart(report['class_probabilities'], horizontal=True)
        if report['prediction_set'] is None:
            st.info('Conformal calibration is unavailable for these weights. No calibrated prediction set is reported.')
        elif report['review_required']:
            st.warning('Review required: the conformal set is empty or contains multiple classes. Set: ' + (', '.join(report['prediction_set']) or 'empty'))
        else:
            st.write('Conformal prediction set: ' + ', '.join(report['prediction_set']))
        st.image(figure, caption='Mask threshold: 0.5 · Grad-CAM targets the predicted class · maps are displayed at 256 × 256 pixels.')
        a, b, c = st.columns(3)
        a.download_button('Download report', json.dumps(report, indent=2), 'brisc_report.json', 'application/json')
        b.download_button('Download mask', mask, 'brisc_mask.png', 'image/png')
        c.download_button('Download visualization', figure, 'brisc_visualization.png', 'image/png')
