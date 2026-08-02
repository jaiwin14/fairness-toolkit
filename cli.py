"""
Convenience entry point: `python cli.py <command>` works directly from the
repo root without needing `pip install -e .` first (though the installed
`fairkit` command, from that same install, is the normal way to use this —
see the README Quickstart).

The real implementation lives in `src/fairkit/cli.py`, as part of the
installable package (needed so the `fairkit` console-script entry point
in pyproject.toml resolves correctly).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fairkit.cli import main

if __name__ == "__main__":
    main()
