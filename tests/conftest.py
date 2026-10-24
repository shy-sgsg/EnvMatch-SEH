from __future__ import annotations

import sys
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/envmatch-matplotlib-cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
