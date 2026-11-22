"""Tables for revision items 3 and 4 + the short summary over items 1-5, from the outputs of revision_items.py and
revision_items_125.py. Writes ONLY to module1_v2/results/revision_items1-4/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
R = V2/'results'/'revision_items1-4'
TABLE6 = ['B0', 'B1', 'Full', 'I-B', 'D3', 'D4', 'D5', 'M0', 'M1', 'M2', 'M3']
CONTROLS = {'BG': 'background, spec placement, B0 pixel rule (skin gate)',
            'BGn': 'background, spec placement, no skin gate',
            'BGm': 'background, mask-only placement, B0 pixel rule (skin gate)',
            'BGmn': 'background, mask-only placement, no skin gate',
            'B0n': 'B0 box, no skin gate (reference for BGn / BGmn)',
            'B0': 'B0 as in the paper (reference for BG / BGm)'}
METRICS = ['usable_window_percent', 'pulse_snr_db', 'informative_window_percent', 'agreement_percent',
           'agreement_testable_percent', 'present_shuffle19_percent', 'present_shuffle199_percent']


def ms(x, d=2):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return f'{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}' if len(x) > 1 else (f'{x[0]:.{d}f}' if len(x) else 'n/a')


def main():
    pv = pd.read_csv(R/'items3_4_per_video_raw.csv')
    place = pd.read_csv(R/'item3_background_placement.csv')
    n_all = pv.video_id.nunique()

    # ---------------- item 3
    rows = []
    for cid, desc in CONTROLS.items():
        part = pv[pv.config == cid]
        if cid in ('BG', 'BGn'):
            placed = place.loc[place.status != 'not_placed', 'video_id']
            scale = place.loc[place.status != 'not_placed', 'scale']
        elif cid in ('BGm', 'BGmn'):
            placed = place.loc[place.maskonly_status != 'not_placed', 'video_id']
            scale = place.loc[place.maskonly_status != 'not_placed', 'maskonly_scale']
        else:
            placed, scale = part.video_id, pd.Series([1.])
        part = part[part.video_id.isin(placed)]
        usable = part[part.usable_window_percent > 0]
        row = dict(config=cid, description=desc, videos_total=n_all, videos_box_placed=len(part),
                   videos_full_size_box=int((scale >= .999).sum()), box_scale_median=float(np.median(scale)),
                   videos_with_usable_trace=len(usable))
        for m in METRICS:    # mean ± SD over the videos where the box was placed (no trace = 0% informative etc.)
            row[m] = ms(part[m], 1 if 'percent' in m else 2)
        row['pulse_snr_db_usable_videos'] = ms(usable.pulse_snr_db)
        rows.append(row)
    item3 = pd.DataFrame(rows)
    item3.to_csv(R/'item3_background_control.csv', index=False)
    # paired: each control vs its own reference on the same videos (descriptive)

    # ---------------- item 4
    rows = []
    for cid in TABLE6 + list(CONTROLS):
        if cid == 'B0' and cid in [r['config'] for r in rows]:
            continue
        part = pv[pv.config == cid]
        d = part.present_shuffle199_percent - part.present_shuffle19_percent
        row = dict(config=cid, videos=len(part), present_19=ms(part.present_shuffle19_percent, 1),
                   present_199=ms(part.present_shuffle199_percent, 1),
                   mean_change_199_minus_19_pp=round(float(d.mean()), 2), null95_db_199=ms(part.null95_db_199))
        if cid == 'Full':
            e = part.present_shuffle199_seed43_percent - part.present_shuffle199_percent
            row.update(present_199_seed43=ms(part.present_shuffle199_seed43_percent, 1),
                       mc_change_mean_pp=round(float(e.mean()), 2),
                       mc_change_abs_per_video_mean_pp=round(float(e.abs().mean()), 2),
                       mc_change_per_video_max_pp=round(float(e.abs().max()), 2))
        rows.append(row)
    item4 = pd.DataFrame(rows)
    item4.to_csv(R/'item4_pulse_present_199.csv', index=False)

    # ---------------- summary
    i1 = pd.read_csv(R/'item1_train_switches_D5_D6.csv').iloc[0]
    i2 = pd.read_csv(R/'item2_D5_paired_tests.csv')
    full = item4.set_index('config').loc['Full']
    s = [dict(item=1, result=f"train (21): D5 {i1.D5_mean_switches_per_sec:.3f}/s, D6 {i1.D6_mean_switches_per_sec:.3f}/s, "
                             f"difference {i1.mean_difference_D6_minus_D5:+.4f}/s -> "
                             f"{'differs' if i1.differs_by_more_than_0_01 else 'does not differ'} by > 0.01/s")]
    for r in i2.itertuples():
        s.append(dict(item=2, result=f"{r.comparison}, {r.metric}: diff {r.mean_difference:+.2f}, r = {r.rank_biserial_r:+.2f}, "
                                     f"Wilcoxon p {r.wilcoxon_p:.3f}, p_Holm {r.p_holm_as_in_paper:.3f} (paper CSV) / "
                                     f"{r.p_holm_exact_ties:.3f} (exact ties)"))
    for r in item3.itertuples():
        s.append(dict(item=3, result=f"{r.config} ({r.videos_box_placed}/{r.videos_total} videos, {r.videos_full_size_box} full size): "
                                     f"SNR {r.pulse_snr_db} dB, informative {r.informative_window_percent}%, "
                                     f"agreement {r.agreement_percent}%, pulse-present(199) {r.present_shuffle_199_percent if hasattr(r, 'present_shuffle_199_percent') else r.present_shuffle199_percent}%"))
    for r in item4.itertuples():
        s.append(dict(item=4, result=f"{r.config}: pulse-present {r.present_19}% (19) -> {r.present_199}% (199)"))
    s.append(dict(item=4, result=f"Full Monte-Carlo: seed 42 {full.present_199}% vs seed 43 {full.present_199_seed43}%; "
                                 f"change {full.mc_change_mean_pp:+.2f} pp (mean |per-video| {full.mc_change_abs_per_video_mean_pp:.2f} pp)"))
    s.append(dict(item=5, result='figures/fig3_stability_D5.png, figures/fig4_snr_D5.png (D5 blue, Full orange)'))
    pd.DataFrame(s).to_csv(R/'summary.csv', index=False)
    print(pd.DataFrame(s).to_string(index=False))


if __name__ == '__main__':
    main()
