"""Video I/O, signal generation and validation helpers for DataAugment.

Reading
    UBFC-rPPG videos are uncompressed AVI files (24-bit BGR DIB, bottom-up rows). OpenCV's FFmpeg decoder
    crashes on these files (``malloc_consolidate(): invalid chunk size``), so ``RawAviReader`` parses the RIFF
    container directly and returns the stored pixels without any decoding step. Other formats fall back to
    ``cv2.VideoCapture``.

Writing
    ``FFV1Writer`` writes lossless FFV1 in a Matroska (.mkv) container through OpenCV's FFmpeg backend and keeps
    an MD5 of every written frame so the file can be verified bit-exact after writing (``verify_output``).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import struct
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

VIDEO_EXTS = ('.avi', '.mp4', '.mkv', '.mov', '.m4v')


# ----------------------------------------------------------------------------------------------------- reading
class RawAviReader:
    """Reader for uncompressed (BI_RGB, 24-bit) AVI files, including OpenDML (RIFF AVIX) extensions."""

    def __init__(self, path):
        self.path = Path(path)
        self.streams, self.movi = [], []
        self._parse()
        vs = [s for s in self.streams if s['type'] == 'vids']
        if not vs:
            raise ValueError(f'{self.path}: no video stream')
        v = vs[0]
        self.video_index = self.streams.index(v)
        bih = v['strf']
        _, w, h, _, bits, comp = struct.unpack('<IiiHHI', bih[:20])
        if comp != 0 or bits != 24:
            raise ValueError(f'{self.path}: not uncompressed 24-bit video (compression={comp}, bits={bits})')
        self.width, self.height = w, abs(h)
        self.bottom_up = h > 0
        self.stride = ((w * 3 + 3) // 4) * 4
        self.frame_bytes = self.stride * self.height
        self.fps_fraction = Fraction(v['rate'], v['scale'])
        self.fps = float(self.fps_fraction)
        self.has_audio = any(s['type'] == 'auds' for s in self.streams)
        self.offsets = self._frame_offsets()
        self.n_frames = len(self.offsets)
        self.duration = self.n_frames / self.fps
        self.codec = 'uncompressed BGR24 (DIB, %s)' % ('bottom-up' if self.bottom_up else 'top-down')

    def _parse(self):
        size = self.path.stat().st_size
        with open(self.path, 'rb') as f:
            pos = 0
            while pos + 12 <= size:
                f.seek(pos)
                cid, sz, typ = struct.unpack('<4sI4s', f.read(12))
                if cid != b'RIFF' or typ not in (b'AVI ', b'AVIX'):
                    break
                self._walk(f, pos + 12, pos + 8 + sz, None)
                pos += 8 + sz + (sz & 1)

    def _walk(self, f, start, end, cur):
        f.seek(start)
        while f.tell() + 8 <= end:
            pos = f.tell()
            cid, sz = struct.unpack('<4sI', f.read(8))
            nxt = pos + 8 + sz + (sz & 1)
            if cid == b'LIST':
                lt = f.read(4)
                if lt == b'movi':
                    self.movi.append((pos + 12, pos + 8 + sz))
                elif lt == b'strl':
                    self.streams.append({'type': None})
                    self._walk(f, pos + 12, pos + 8 + sz, self.streams[-1])
                else:
                    self._walk(f, pos + 12, pos + 8 + sz, cur)
            elif cid == b'strh' and cur is not None:
                d = f.read(sz)
                cur['type'] = d[:4].decode('latin1')
                cur['scale'], cur['rate'], _, cur['length'] = struct.unpack('<IIII', d[20:36])
            elif cid == b'strf' and cur is not None:
                cur['strf'] = f.read(sz)
            f.seek(nxt)

    def _frame_offsets(self):
        tags = {b'%02ddb' % self.video_index, b'%02ddc' % self.video_index}
        offs = []
        with open(self.path, 'rb') as f:
            for start, end in self.movi:
                f.seek(start)
                while f.tell() + 8 <= end:
                    pos = f.tell()
                    cid, sz = struct.unpack('<4sI', f.read(8))
                    if cid == b'LIST':          # 'rec ' lists inside movi
                        f.seek(pos + 12)
                        continue
                    if cid in tags and sz == self.frame_bytes:
                        offs.append(pos + 8)
                    f.seek(pos + 8 + sz + (sz & 1))
        return offs

    def _decode(self, buf):
        a = np.frombuffer(buf, np.uint8).reshape(self.height, self.stride)[:, :self.width * 3]
        a = a.reshape(self.height, self.width, 3)
        return np.ascontiguousarray(a[::-1] if self.bottom_up else a)

    def read(self, i):
        with open(self.path, 'rb') as f:
            f.seek(self.offsets[i])
            return self._decode(f.read(self.frame_bytes))

    def frames(self, max_frames=None):
        n = self.n_frames if max_frames is None else min(self.n_frames, max_frames)
        with open(self.path, 'rb') as f:
            for i in range(n):
                f.seek(self.offsets[i])
                yield self._decode(f.read(self.frame_bytes))


class CvReader:
    """Fallback reader for compressed formats (decoding through OpenCV/FFmpeg)."""

    def __init__(self, path):
        self.path = Path(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f'cannot open {path}')
        self.width, self.height = int(cap.get(3)), int(cap.get(4))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.fps_fraction = Fraction(self.fps).limit_denominator(1001000)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.n_frames / self.fps if self.fps else float('nan')
        self.has_audio = None           # unknown without ffprobe
        self.codec = 'decoded by OpenCV/FFmpeg'
        cap.release()

    def read(self, i):
        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        cap.release()
        return f if ok else None

    def frames(self, max_frames=None):
        cap = cv2.VideoCapture(str(self.path))
        i = 0
        while max_frames is None or i < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            i += 1
            yield f
        cap.release()


def open_video(path):
    try:
        return RawAviReader(path)
    except (ValueError, struct.error):
        return CvReader(path)


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(p))]


def find_videos(root, n):
    """First ``n`` videos under ``root`` in natural order (subject1, subject3, subject4, ...)."""
    files = [p for p in Path(root).rglob('*') if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    return sorted(files, key=natural_key)[:n]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------- writing
class FFV1Writer:
    """Lossless FFV1/MKV writer that records the MD5 of every frame it writes."""

    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'FFV1'), float(fps), tuple(size), True)
        if not self.w.isOpened():
            raise IOError(f'OpenCV could not open an FFV1 writer for {self.path}')
        self.size = tuple(size)
        self.md5 = []

    def write(self, frame):
        assert frame.dtype == np.uint8 and frame.shape[1::-1] == self.size, 'frame size/dtype changed'
        frame = np.ascontiguousarray(frame)
        self.md5.append(hashlib.md5(frame.tobytes()).hexdigest())
        self.w.write(frame)

    def close(self):
        self.w.release()


def verify_output(path, expected_md5):
    """Decode the written file and compare every frame with what was written."""
    cap = cv2.VideoCapture(str(path))
    info = {'width': int(cap.get(3)), 'height': int(cap.get(4)), 'fps_container': float(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count_header': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    n, mism = 0, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if n >= len(expected_md5) or hashlib.md5(np.ascontiguousarray(f).tobytes()).hexdigest() != expected_md5[n]:
            mism.append(n)
        n += 1
    cap.release()
    info.update(frame_count_decoded=n, frames_mismatched=len(mism) + max(0, len(expected_md5) - n),
                first_mismatches=mism[:10],
                lossless_bit_exact=(n == len(expected_md5) and not mism),
                file_size_mb=round(os.path.getsize(path) / 2 ** 20, 1))
    return info


def mux_audio(video_path, source_path):
    """Copy the source audio stream into the output (only if an ffmpeg binary exists)."""
    ff = shutil.which('ffmpeg')
    if ff is None:
        return {'audio_preserved': False, 'reason': 'source has audio but no ffmpeg binary is available'}
    tmp = Path(video_path).with_suffix('.tmp.mkv')
    cmd = [ff, '-y', '-v', 'error', '-i', str(video_path), '-i', str(source_path), '-map', '0:v:0', '-map', '1:a?',
           '-c', 'copy', str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {'audio_preserved': False, 'reason': r.stderr.strip()[:300]}
    os.replace(tmp, video_path)
    return {'audio_preserved': True, 'method': 'stream copy (no re-encoding), same start time'}


# ----------------------------------------------------------------------------------------- subject localization
def detect_subject_box(reader, n_probe=10, step=10):
    """Median Haar-cascade face box over the first frames; used to place occluders and shadows on the subject."""
    casc = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
    boxes = []
    for i in range(0, min(reader.n_frames, n_probe * step), step):
        g = cv2.cvtColor(reader.read(i), cv2.COLOR_BGR2GRAY)
        f = casc.detectMultiScale(g, 1.1, 5, minSize=(g.shape[0] // 8, g.shape[0] // 8))
        if len(f):
            boxes.append(max(f, key=lambda b: b[2] * b[3]))
    if boxes:
        x, y, w, h = np.median(np.array(boxes), axis=0)
        return {'x': float(x), 'y': float(y), 'w': float(w), 'h': float(h), 'method': 'haar_frontalface',
                'detections': len(boxes)}
    W, H = reader.width, reader.height
    return {'x': W * 0.35, 'y': H * 0.2, 'w': W * 0.3, 'h': H * 0.35, 'method': 'fallback_center', 'detections': 0}


# ------------------------------------------------------------------------------------- temporal control signals
def smooth_noise(n, fps, cutoff_hz, rng, order=4, pad_s=30.0):
    """Aperiodic, zero-phase low-pass filtered Gaussian noise; zero mean, max |x| = 1.

    No sinusoids are used, so the signal has no spectral line; power above ``cutoff_hz`` falls off at
    2*order*20 dB/decade (zero-phase filtering doubles the order).
    """
    pad = int(round(pad_s * fps))
    x = rng.standard_normal(n + 2 * pad)
    sos = signal.butter(order, cutoff_hz, 'low', fs=fps, output='sos')
    y = signal.sosfiltfilt(sos, x)[pad:pad + n]
    y -= y.mean()
    m = np.abs(y).max()
    return y / m if m > 0 else y


def bandlimited_noise(n, fps, lo, hi, rng, order=4, pad_s=5.0):
    """Zero-phase band-pass filtered Gaussian noise, unit RMS."""
    pad = int(round(pad_s * fps))
    hi = min(hi, 0.45 * fps)
    sos = signal.butter(order, [lo, hi], 'band', fs=fps, output='sos')
    y = signal.sosfiltfilt(sos, rng.standard_normal(n + 2 * pad))[pad:pad + n]
    s = y.std()
    return y / s if s > 0 else y


def spectral_check(x, fps, bands, power_thresh=0.01, peak_thresh=10.0, min_seconds=30.0, local_hz=0.25):
    """Share of (detrended) signal power in physiological bands and whether a narrow spectral line exists there.

    Welch spectrum (20-s segments, 0.05 Hz resolution). For every bin in a band the local prominence is the bin
    power divided by the running median over a symmetric window of +-``local_hz`` (shrunk near the spectrum
    edges). A monotonic (e.g. steeply falling low-pass) spectrum gives a prominence of 1, whereas a spectral
    line stands out of the median. A band
    is flagged only if it holds > ``power_thresh`` of the total power AND its largest local prominence exceeds
    ``peak_thresh``, i.e. a respiration- or pulse-like line. Broadband noise and smooth (falling) spectra pass.
    Signals shorter than ``min_seconds`` are not assessed (insufficient frequency resolution).
    """
    x = np.asarray(x, float)
    if len(x) < min_seconds * fps:
        return {'assessed': False, 'reason': f'shorter than {min_seconds:.0f} s'}
    if np.ptp(x) < 1e-12:
        return {'assessed': True, 'constant': True}
    x = signal.detrend(x)
    nper = int(round(fps * 20))
    f, P = signal.welch(x, fs=fps, nperseg=nper, noverlap=nper // 2)
    tot = P[f > 0].sum()
    df = f[1] - f[0]
    w = max(3, int(round(local_hz / df)))
    out = {'assessed': True}
    for name, (lo, hi) in bands.items():
        idx = np.where((f >= lo) & (f <= hi))[0]
        prom = []
        for k in idx:
            h = min(w, k - 1, len(P) - 1 - k)          # symmetric window, so a monotonic spectrum gives 1
            prom.append(P[k] / max(np.median(P[k - h:k + h + 1]), 1e-30))
        prom = np.array(prom)
        frac = float(P[idx].sum() / tot) if tot > 0 else 0.0
        j = int(np.argmax(prom))
        out[name] = {'power_fraction': round(frac, 6), 'max_local_prominence': round(float(prom[j]), 2),
                     'at_hz': round(float(f[idx[j]]), 3),
                     'narrowband_flag': bool(frac > power_thresh and prom[j] > peak_thresh)}
    return out


def step_stats(x):
    d = np.abs(np.diff(np.asarray(x, float)))
    return {'max_abs_step': round(float(d.max()), 5) if len(d) else 0.0,
            'median_abs_step': round(float(np.median(d)), 5) if len(d) else 0.0}


class FrameDiffTracker:
    """Mean absolute difference between consecutive frames (grey, 1/4 size) to detect temporal discontinuities."""

    def __init__(self):
        self.prev, self.d = None, []

    def add(self, frame):
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.prev is not None:
            self.d.append(float(np.abs(g - self.prev).mean()))
        self.prev = g

    def summary(self):
        d = np.array(self.d) if self.d else np.zeros(1)
        med = float(np.median(d))
        return {'median': round(med, 4), 'p99': round(float(np.percentile(d, 99)), 4), 'max': round(float(d.max()), 4),
                'spike_ratio_max_over_median': round(float(d.max() / med), 2) if med > 0 else None}
