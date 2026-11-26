# models/

Place `brisc2025_bundle.joblib` here before running `predict.py`.

## How to obtain the bundle

### Option A – Download from Google Drive (recommended)
After running the Colab export cell (see `README.md → Step 1`), download
`brisc2025_bundle.joblib` from your Google Drive and move it here:

```bash
mv ~/Downloads/brisc2025_bundle.joblib models/
```

### Option B – Download from GitHub Releases
If the bundle was uploaded as a GitHub Release asset, grab it with:

```bash
gh release download v1.0.0 --pattern "*.joblib" --dir models/
```

## Why is the file not in git?

The compressed bundle is ~65–75 MB.  Files under 95 MB can be committed
directly; files above that threshold should be attached to a GitHub Release
(which supports up to 2 GB per asset).  See the root `.gitignore` for the
current policy.
