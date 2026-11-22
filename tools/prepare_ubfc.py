"""UBFC-rPPG preparation (from the original UBFC check notebook, cell 4; same code).

1. Lossless, bit-exact FFV1 copy of subjects 1, 3 and 4: data/ubfc/subjectN/vid.avi -> data/ubfc_flat/subjectN.avi.
   The raw UBFC AVIs (uncompressed bottom-up DIB frames) crash OpenCV 4.14's reader, so the RIFF chunks are read directly.
2. Reference HR: FFT peak (45-180 bpm) of the fingertip PPG waveform in 8-s windows with a 1-s step, plus the
   oximeter HR: HR RR using module1 v2/runs/ubfc_flat_38d8f553/ubfc_reference/reference_hr_{ppg,oximeter}.csv.
The run name ubfc_flat_38d8f553 is the one used in the paper (the original notebook derived it from the file sizes).
"""
import hashlib
import os
import struct
import sys

import cv2
import numpy as np
import pandas as pd

from common import ROOT

SUBJECTS = ('subject1', 'subject3', 'subject4')
RUN_NAME = 'ubfc_flat_38d8f553'
LOW, HIGH = 45.0, 180.0


def raw_avi_frames(path):
    """Yield (fps, BGR frame) from an uncompressed 24-bit DIB AVI by reading the RIFF chunks directly."""
    size_total = os.path.getsize(path)
    with open(path, 'rb') as fh:
        assert fh.read(4) == b'RIFF'
        fh.read(4)
        assert fh.read(4) == b'AVI ', 'not an AVI file'
        state = dict(w=None, h=None, fps=None, bottom_up=True)

        def walk(end):
            while fh.tell() + 8 <= end:
                cid = fh.read(4)
                size = struct.unpack('<I', fh.read(4))[0]
                start = fh.tell()
                if cid in (b'LIST', b'RIFF'):
                    fh.read(4)
                    yield from walk(start + size)
                elif cid == b'strh' and state['fps'] is None:
                    d = fh.read(size)
                    if d[:4] == b'vids':
                        scale, rate = struct.unpack('<II', d[20:28])
                        state['fps'] = rate / scale
                elif cid == b'strf' and state['w'] is None:
                    d = fh.read(size)
                    w, h, _, bits = struct.unpack('<iiHH', d[4:16])
                    assert bits == 24, f'expected 24-bit frames, got {bits}'
                    state.update(w=w, h=abs(h), bottom_up=h > 0)
                elif cid[2:] in (b'db', b'dc') and state['w'] and size == state['w'] * state['h'] * 3:
                    a = np.frombuffer(fh.read(size), np.uint8).reshape(state['h'], state['w'], 3)
                    yield state['fps'], np.ascontiguousarray(a[::-1] if state['bottom_up'] else a)
                fh.seek(start + size + (size & 1))
        yield from walk(size_total)


def ppg_hr(ppg, t, centre, half=4.0):
    m = (t >= centre - half) & (t < centre + half)
    if m.sum() < 64:
        return np.nan
    x, tt = ppg[m], t[m]
    fs = (len(tt) - 1) / (tt[-1] - tt[0])
    grid = np.linspace(tt[0], tt[-1], len(tt))
    x = np.interp(grid, tt, x)
    x = x - x.mean()
    nfft = 1 << 16
    f = np.fft.rfftfreq(nfft, 1 / fs) * 60
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft))
    band = (f >= LOW) & (f <= HIGH)
    return float(f[band][np.argmax(p[band])])


def main():
    src, flat = ROOT/'data/ubfc', ROOT/'data/ubfc_flat'
    flat.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in SUBJECTS:
        video, target = src/s/'vid.avi', flat/f'{s}.avi'
        if not video.is_file():
            sys.exit(f'{video} not found: run tools/setup_paths.py with ubfc_rppg configured')
        if not target.exists():
            tmp, writer, n = target.with_suffix('.tmp.avi'), None, 0
            for fps, frame in raw_avi_frames(video):
                if writer is None:
                    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'FFV1'), fps, (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                n += 1
            writer.release()
            tmp.rename(target)
            print(f'{s}: {n} frames -> {target.relative_to(ROOT)} (lossless FFV1)')
        cap = cv2.VideoCapture(str(target))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        rows.append(dict(subject=s, video_id='NEW_' + s, fps=fps, frames=n, duration_sec=n / fps))
    ref_dir = ROOT/'HR RR using module1 v2/runs'/RUN_NAME/'ubfc_reference'
    ref_dir.mkdir(parents=True, exist_ok=True)
    oxi, ppgref = [], []
    for r in rows:
        vals = [np.array(line.split(), float) for line in (src/r['subject']/'ground_truth.txt').read_text().strip().splitlines()]
        ppg, hr_line, t = vals[0], vals[1], vals[2]
        if t[-1] > 5 * r['duration_sec']:                 # timestamps in ms in some copies
            t = t / 1000.
        t = t - t[0]
        oxi.append(pd.DataFrame(dict(video_id=r['video_id'], timestamp_sec=t, hr_bpm=hr_line)))
        centres = np.arange(4.0, t[-1] - 4.0 + 1e-9, 1.0)
        ppgref.append(pd.DataFrame(dict(video_id=r['video_id'], timestamp_sec=centres,
                                        hr_bpm=[ppg_hr(ppg, t, c) for c in centres])))
    pd.concat(oxi).to_csv(ref_dir/'reference_hr_oximeter.csv', index=False)
    ref = pd.concat(ppgref)
    ref.to_csv(ref_dir/'reference_hr_ppg.csv', index=False)
    # the same reference rows are part of the released face_track_baseline reference: they must agree
    rel = pd.read_csv(ROOT/'released/face_track_baseline/reference_hr_ppg.csv')
    m = ref.merge(rel, on=['video_id', 'timestamp_sec'], suffixes=('', '_released'))
    diff = float((m.hr_bpm - m.hr_bpm_released).abs().max())
    print(f'reference HR: {len(ref)} windows; compared with the released reference on {len(m)} windows, '
          f'max |difference| = {diff:.2e} bpm')
    if len(m) == 0 or diff > 1e-6:
        sys.exit('reference HR differs from the released one: are these the UBFC-rPPG DATASET_2 files?')


if __name__ == '__main__':
    main()
