"""Reference HR for the HR evaluation: the UBFC PPG-waveform HR per 8 s window (1 s step).

The values for the original videos (NEW_subject1/3/4) were computed by the UBFC check notebook (FFT peak of the
fingertip PPG waveform in the same 8 s / 1 s windows as the camera HR). Augmentation keeps every frame and its
timestamp, so an augmented video has exactly the reference of its source subject; the rows are copied under the
augmented video IDs (NEW_<subject>_<category>_<severity>).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT/'HR RR using module1 v2/runs/ubfc_flat_38d8f553/ubfc_reference/reference_hr_ppg.csv'
OUT = Path(__file__).resolve().parent/'reference_hr_ppg.csv'


def main():
    ref = pd.read_csv(SRC)
    parts = [ref]
    for video in sorted(Path(__file__).resolve().parent.glob('videos*/*.mkv')):
        subject = video.stem.split('_')[0]                     # subject1_motion_small -> subject1
        r = ref[ref.video_id == f'NEW_{subject}'].copy()
        if r.empty:
            raise SystemExit(f'no reference for {subject}')
        r['video_id'] = f'NEW_{video.stem}'
        parts.append(r)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(OUT, len(out), 'rows,', out.video_id.nunique(), 'videos')


if __name__ == '__main__':
    main()
