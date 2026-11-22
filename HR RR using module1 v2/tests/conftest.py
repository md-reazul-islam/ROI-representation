import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/'HR RR using module1 v2/src', ROOT/'module1_v2/src', ROOT/'src', ROOT/'scripts'):
    sys.path.insert(0, str(p))
