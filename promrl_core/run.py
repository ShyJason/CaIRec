#!/usr/bin/env python3

import os
from pathlib import Path
import runpy
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
entrypoint_env = os.environ.get("PROMRL_ENTRYPOINT")

if not entrypoint_env:
    raise SystemExit(
        "promrl_core/run.py is for the original ProMRL training CLI, but this "
        "MMRec checkout only vendors the ProMRL Python modules used by model.py. "
        "The original ProMRL top-level training entrypoint was not present in "
        "the vendored source. Set PROMRL_ENTRYPOINT=/path/to/original/run.py "
        "if you need to run the standalone ProMRL scripts."
    )

ENTRYPOINT = Path(entrypoint_env).expanduser().resolve()
if not ENTRYPOINT.exists():
    raise SystemExit(f"PROMRL_ENTRYPOINT does not exist: {ENTRYPOINT}")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

runpy.run_path(str(ENTRYPOINT), run_name="__main__")
