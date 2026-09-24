# Build the documentation

From the repository root, with Python 3.12:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server --bind 127.0.0.1 --directory docs/_build/html
```

Open http://localhost:8000. API pages read the package's docstrings; the editable
install keeps them linked to this checkout. Rebuild after changing docstrings.
The getting-started page includes the root README. The conformal and OT pages
include their package READMEs; `usage.md` connects the mathematical construction
to the API.
