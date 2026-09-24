"""Importing the package leaves Numba unloaded."""

import subprocess
import sys


def test_import_is_lightweight():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import yemale; assert 'numba' not in sys.modules",
        ],
        check=True,
    )
