"""Run each document's Python examples in reading order."""

import re
from pathlib import Path

import pytest


@pytest.mark.parametrize("filename", ["README.md", "docs/usage.md"])
def test_document_examples(filename):
    path = Path(__file__).resolve().parents[1] / filename
    document = path.read_text(encoding="utf-8")
    blocks = list(
        re.finditer(r"^```python\n(.*?)^```$", document, re.MULTILINE | re.DOTALL)
    )
    assert blocks, f"No Python examples in {filename}"
    namespace = {}
    for block in blocks:
        padding = "\n" * document.count("\n", 0, block.start(1))
        exec(compile(padding + block.group(1), str(path), "exec"), namespace)  # noqa: S102
