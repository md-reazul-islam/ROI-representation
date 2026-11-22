"""Download the four model weights into models/ and verify their SHA-256 (the files used for the paper).

    python tools/download_models.py
Sapiens2 needs the Hugging Face CLI (pip install huggingface_hub) and acceptance of its licence on huggingface.co.
"""
import hashlib
import shutil
import subprocess
import sys
import urllib.request

from common import ROOT
from setup_paths import MODELS

M = ROOT/'models'
SAM2_URL = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    M.mkdir(exist_ok=True)
    for name in ('yolo11s.pt', 'rtdetr-l.pt'):
        if not (M/name).is_file():
            from ultralytics.utils.downloads import attempt_download_asset
            got = attempt_download_asset(name)          # official Ultralytics release asset
            shutil.move(got, M/name)
    if not (M/'sam2.1_hiera_tiny.pt').is_file():
        urllib.request.urlretrieve(SAM2_URL, M/'sam2.1_hiera_tiny.pt')
    if not (M/'sapiens2_0.4b_pose.safetensors').is_file():
        subprocess.run(['hf', 'download', 'facebook/sapiens2-pose-0.4b', 'sapiens2_0.4b_pose.safetensors',
                        '--local-dir', str(M)], check=True)
    bad = False
    for name, digest in MODELS.items():
        ok = sha256(M/name) == digest
        bad |= not ok
        print(f'{name:34s} {"OK" if ok else "SHA-256 DIFFERS from the file used in the paper"}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
