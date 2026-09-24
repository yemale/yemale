"""Run documentation examples and check API-to-source navigation."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "filename",
    [
        "README.md",
        "src/yemale/conformal/README.md",
        "src/yemale/ot/README.md",
        "docs/usage.md",
    ],
)
def test_document_examples(filename):
    path = Path(__file__).resolve().parents[1] / filename
    document = path.read_text(encoding="utf-8")
    blocks = list(
        re.finditer(r"^```python\n(.*?)^```$", document, re.MULTILINE | re.DOTALL)
    )
    assert blocks, f"No Python examples in {filename}"
    namespace = {}
    if filename in ("README.md", "src/yemale/conformal/README.md"):
        # The model examples start with an existing predictor and fresh data.
        class LinearModel:
            def predict(self, inputs):
                return np.asarray(inputs) @ np.array([1.0, 2.0])

        model = LinearModel()
        rng = np.random.default_rng(0)
        X_cal, X_new = rng.normal(size=(99, 2)), rng.normal(size=(3, 2))
        namespace.update(
            model=model,
            X_cal=X_cal,
            y_cal=model.predict(X_cal) + rng.normal(size=99),
            X_new=X_new,
            y_new=model.predict(X_new) + rng.normal(size=3),
        )
    for block in blocks:
        # Keep traceback line numbers aligned with the document.
        padding = "\n" * document.count("\n", 0, block.start(1))
        exec(compile(padding + block.group(1), str(path), "exec"), namespace)  # noqa: S102


def test_help_names_the_default_score(capsys):
    import yemale

    help(yemale.extend)
    output = capsys.readouterr().out
    assert "score=residual" in output
    assert "<function" not in output


def test_conformal_source_backlinks(tmp_path_factory):
    pytest.importorskip("sphinx")
    from bs4 import BeautifulSoup

    root = Path(__file__).resolve().parents[1]
    html = tmp_path_factory.getbasetemp() / "html"
    command = [sys.executable, "-m", "sphinx", "-W", "--keep-going"]
    command.extend(["-b", "html", str(root / "docs"), str(html)])
    subprocess.run(command, check=True)
    source = BeautifulSoup(
        (html / "_modules/yemale/conformal/predict.html").read_text(encoding="utf-8"),
        "html.parser",
    )
    page = BeautifulSoup(
        (html / "conformal.html").read_text(encoding="utf-8"), "html.parser"
    )
    # Our viewcode hook restores the public conformal namespace.
    for name in ("CPD", "CPD.mean"):
        link = source.find(id=name).find("a", class_="viewcode-back")["href"]
        anchor = f"yemale.conformal.{name}"
        assert link.endswith(f"conformal.html#{anchor}")
        assert page.find(id=anchor)
