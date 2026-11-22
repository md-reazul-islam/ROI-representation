"""D8 rule R2 input: per sampled frame and candidate region, the mean colour of ALL pixels inside the region polygon.

    python module1_v2_d8/scripts/colour_features.py --run runs/ubfc_aug

Polygons are the D5 export's `pixel_polygon` (module1_v2/runs/<run>/dynamic_roi_v2_D5/dynamic_roi_v2.csv, read only), so
D8 sees exactly the pixels D5 uses. Frames are read sequentially with OpenCV, as the Module 1 pipeline does.
Output: module1_v2_d8/runs/<run>/colour_features.csv with
  video_id, frame_index, roi_name, n_px, rn, gn, lum     (rn = R/(R+G+B), gn = G/(R+G+B), lum = mean(R+G+B)/3, 0-255)
Nothing outside module1_v2_d8/ is written.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

D8 = Path(__file__).resolve().parents[1]
ROOT = D8.parent
MIN_PX = 50


def region_colour(frame, polygon):
    """(n_px, rn, gn, lum) of the pixels inside the polygon, or None."""
    if not polygon:
        return None
    p = np.asarray(polygon, np.float32)
    if p.ndim != 2 or len(p) < 3:
        return None
    H, W = frame.shape[:2]
    x0, y0 = np.floor(p.min(0)).astype(int)
    x1, y1 = np.ceil(p.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.round(p - [x0, y0]).astype(np.int32)], 1)
    n = int(mask.sum())
    if n < MIN_PX:
        return None
    crop = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)[mask.ravel() > 0]
    b, g, r = crop.mean(0)
    s = r + g + b
    if s <= 0:
        return None
    return n, r / s, g / s, s / 3.


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    a = ap.parse_args(argv)
    run = (ROOT/a.run).resolve()
    src = ROOT/'module1_v2/runs'/run.name/'dynamic_roi_v2_D5'/'dynamic_roi_v2.csv'
    out_dir = D8/'runs'/run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir/'colour_features.csv'
    table = pd.read_csv(src, usecols=['video_id', 'frame_index', 'roi_name', 'available', 'pixel_polygon'])
    manifest = pd.read_csv(run/'manifest.csv').set_index('video_id')
    rows = []
    for video, g in table.groupby('video_id'):
        polys = {}
        for f, r, ok, p in zip(g.frame_index, g.roi_name, g.available, g.pixel_polygon):
            if ok and isinstance(p, str) and p != 'null':
                polys.setdefault(int(f), []).append((r, json.loads(p)))
        wanted = sorted(polys)
        cap = cv2.VideoCapture(str(manifest.loc[video, 'video_path']))
        i, k, mismatch = 0, 0, 0
        while k < len(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if i == 0 and (frame.shape[1], frame.shape[0]) != (int(manifest.loc[video, 'width']), int(manifest.loc[video, 'height'])):
                mismatch = 1
            if i == wanted[k]:
                for roi, poly in polys[i]:
                    c = region_colour(frame, poly)
                    if c:
                        rows.append((video, i, roi) + c)
                k += 1
            i += 1
        cap.release()
        print(f'{video}: {k}/{len(wanted)} sampled frames read' + ('  WARNING: frame size differs from manifest' if mismatch else ''),
              flush=True)
    pd.DataFrame(rows, columns=['video_id', 'frame_index', 'roi_name', 'n_px', 'rn', 'gn', 'lum']).to_csv(target, index=False)
    print('wrote', target, len(rows), 'rows')


if __name__ == '__main__':
    main()
