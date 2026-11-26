# Review and push to GitHub

Run commands from `~/projects/brisc2025-trustworthy-multitask-mri`.
The accidental trailing newline in the local folder name has been removed.
The existing Git history and origin remote have been preserved.

```bash
python -m pytest -q
git status --short
git diff --check
git diff
```

Review new files too. Then create a branch and commit the source:

```bash
git switch -c feature/mri-interface
git add README.md requirements.txt pyproject.toml LICENSE .gitignore .github app.py brisc_mri tests docs notebooks checkpoints/README.md
git diff --cached --stat
git commit -m "Add MRI interface and modular training pipeline"
```

Authenticate using the GitHub CLI or your existing credential helper. Do not put
personal access tokens into notebook cells, Git remote URLs or committed files.

```bash
gh auth login
git push -u origin feature/mri-interface
```

Open a pull request into main after CI passes. Uploading source does not deploy
the interface automatically. Locally run `streamlit run app.py`; Streamlit
hosting needs the app entry point plus model weights supplied separately.

Raw `.pt` / `.ckpt` model binaries are ignored by Git and can exceed GitHub's regular file-size limit; attach large `.pt` checkpoints to a GitHub Release or use a model hosting service. The compressed CPU bundle `models/brisc2025_bundle.joblib` (88 MB, `compress=3`) is under GitHub's 100 MB limit and is tracked directly in `models/`. Include checksums and training/calibration provenance when sharing weights.

The portable notebook can clone these changes from main after they are merged.
For a feature branch, clone/check out that branch explicitly in Colab instead.
