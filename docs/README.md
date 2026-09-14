# Build the documentation

From the repository root, with Python 3.12:

```bash
python -m pip install ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server --bind 127.0.0.1 --directory docs/_build/html
```

Open http://localhost:8000. API pages read the installed package's docstrings;
the getting-started page includes the README, and the guide uses `usage.md`.
