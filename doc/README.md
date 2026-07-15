# blueqat documentation

Built with Sphinx (furo theme). English only.

## Build locally

```
pip install -e ..[dev]
pip install -r requirements.txt
sphinx-build -b html source build/html
open build/html/index.html
```

Docs are deployed to GitHub Pages automatically on every push to `main`
(see `.github/workflows/docs.yml`).
