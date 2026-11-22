import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for path in (V2/'src', ROOT/'src', ROOT/'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
