from __future__ import annotations

import sys
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/envmatch-matplotlib-cache")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
