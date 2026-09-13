"""Importing the package must not load a compiler or accelerator backend."""

import subprocess
import sys


def test_import_is_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import yemale; "
                "assert not {'numba', 'llvmlite', 'jax', 'torch', 'tensorflow', 'cupy'} "
                "& sys.modules.keys()"
            ),
        ],
        check=True,
    )
