"""Connect the configured dataset folders to the locations the code expects (symbolic links), and check them.

    python tools/setup_paths.py            # link and check
    python tools/setup_paths.py --check    # check only

The code was written for a fixed layout (data/final34/dataset, data/ubfc/subjectN, dataaugment/data, models, ...).
Instead of editing the code, every such location becomes a link to the folder given in config/paths.yaml.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

from common import ROOT, paths

UBFC_SUBJECTS = ('subject1', 'subject3', 'subject4')
MODELS = {'yolo11s.pt': '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5',
          'rtdetr-l.pt': '6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f',
          'sam2.1_hiera_tiny.pt': '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69',
          'sapiens2_0.4b_pose.safetensors': 'f1b7b0e5423d667634a355dd86a14c5b03d243f2ac77943301c7db9f93329106'}
# location used by the code -> (config key, sub-folder of that location)
DERIVED = {'data/ubfc_flat': 'ubfc_flat', 'module1_v2_d8/data_swaddle': 'd8_swaddle',
           'module1_v2_d9/data_swaddle_test': 'd9_swaddle_test', 'module1_v2_d9/swaddle_motion/data': 'swaddle_motion',
           'occlusion_position_test/data': 'occlusion_positions'}
AUGMENT = ('data', 'data_motion_fast', 'data_motion_fast_xl')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def link(where, target, check):
    where = ROOT/where
    target = Path(target)
    if check:
        return
    target.mkdir(parents=True, exist_ok=True)
    if where.is_symlink():
        if Path(os.readlink(where)) == target:
            return
        where.unlink()
    elif where.exists():
        if where.is_dir() and not any(where.iterdir()):
            where.rmdir()
        else:
            print(f'  kept existing {where.relative_to(ROOT)} (not a link; remove it to use {target})')
            return
    where.parent.mkdir(parents=True, exist_ok=True)
    where.symlink_to(target, target_is_directory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='only report, do not create links')
    ap.add_argument('--hash', action='store_true', help='also verify the SHA-256 of every neonatal video (slow)')
    args = ap.parse_args()
    P = paths()
    status = {}

    # 1. neonatal videos -> data/final34/dataset, and the manifest's video_path
    man_path = ROOT/'runs/final34_paper/manifest.csv'
    man = pd.read_csv(man_path)
    nv = P['neonatal_videos']
    if nv and nv.is_dir():
        missing = [f for f in man.filename if not (nv/f).is_file()]
        if missing:
            status['neonatal videos'] = f'MISSING {len(missing)} of {len(man)} files, e.g. {missing[:3]}'
        else:
            if args.hash:
                bad = [f for f, h in zip(man.filename, man.sha256) if sha256(nv/f) != h]
                if bad:
                    sys.exit(f'SHA-256 differs for {bad}: these are not the videos of the paper')
            link('data/final34/dataset', nv, args.check)
            if not args.check:
                man['video_path'] = [str((ROOT/'data/final34/dataset'/f)) for f in man.filename]
                man.to_csv(man_path, index=False)
            status['neonatal videos'] = f'OK ({len(man)} videos{", hashes verified" if args.hash else ""})'
    else:
        status['neonatal videos'] = 'not configured (Level C is skipped)'

    # 2. UBFC-rPPG subjects 1, 3, 4 -> data/ubfc/subjectN
    ub = P['ubfc_rppg']
    if ub and ub.is_dir():
        miss = [s for s in UBFC_SUBJECTS if not ((ub/s/'vid.avi').is_file() and (ub/s/'ground_truth.txt').is_file())]
        if miss:
            status['UBFC-rPPG'] = f'MISSING {miss} (need <ubfc_rppg>/subjectN/vid.avi and ground_truth.txt)'
        else:
            for s in UBFC_SUBJECTS:          # only these three: the augmentation assigns conditions in subject order
                link(f'data/ubfc/{s}', ub/s, args.check)
            status['UBFC-rPPG'] = 'OK (subjects 1, 3, 4)'
    else:
        status['UBFC-rPPG'] = 'not configured (Level B is skipped)'

    # 3. DataAugment videos and other derived videos
    da = P['dataaugment_videos']
    if da:
        for sub in AUGMENT:
            link(f'dataaugment/{sub}', da/sub, args.check)
        n = sum(1 for _ in da.rglob('*.mkv')) if da.is_dir() else 0
        status['DataAugment videos'] = f'{da} ({n} videos present)'
    dv = P['derived_videos']
    if dv:
        for where, sub in DERIVED.items():
            link(where, dv/sub, args.check)
        status['derived videos'] = str(dv)

    # 4. model weights -> models/
    md = P['models']
    if md and md.resolve() != (ROOT/'models').resolve():
        link('models', md, args.check)
    have = {m: (ROOT/'models'/m).is_file() for m in MODELS}
    status['model weights'] = 'OK' if all(have.values()) else f'missing {[m for m, h in have.items() if not h]} (Step 3)'

    width = max(map(len, status))
    for k, v in status.items():
        print(f'{k:<{width}}  {v}')


if __name__ == '__main__':
    main()
