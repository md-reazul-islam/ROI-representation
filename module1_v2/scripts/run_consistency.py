"""Internal-consistency experiments for the corrected condition terms and the stabilized selection (label-free).

E1 Repeatability (test-retest over consecutive sampled frames, within ROI):
   R = 1 - E[(x_t - x_{t+1})^2] / (2 Var x) for every condition term and score, and the Kendall rank agreement of
   the 17-region ranking between consecutive frames, plus the top-1 vs top-2 score margin (near-ties).
E2 Score perturbation: Gaussian noise (sigma) added to every ROI-frame score; agreement of the perturbed selection
   with the unperturbed one (per-frame Jaccard of the selected set, top-region agreement) and switch rate.
E3 Definition sensitivity of V': keypoint-confidence threshold 0.3-0.7 (V' recomputed from the stored keypoints);
   agreement of V'_theta with V'_0.5 and of the D5 selection.
E4 Estimator sensitivity of M_n: global motion by background median (default) vs RANSAC homography; agreement of
   M_n and of the D6 selection (needs features/roi_features_homography.csv).
Selection-level only: no pixels or pulses are recomputed. Video = statistical unit.
Usage: python module1_v2/scripts/run_consistency.py --run runs/final34_paper
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for p in (V2/'src', ROOT/'src', ROOT/'scripts', V2/'scripts'):
    sys.path.insert(0, str(p))
import run_v2_evaluation as ev                                   # noqa: E402
from module1.roi.anatomical import AnatomicalROIMapper           # noqa: E402
from module2.ablation import bootstrap_ci, holm, paired_test     # noqa: E402
from module1_v2 import selection as sel, terms                   # noqa: E402

TERMS = dict(P='pose_term', V='visibility_term', V_prime='visibility_keypoint', S='skin_term', C='temporal_term',
             M='motion_term', M_n='motion_nuisance', Q='Q_v1', Q_prime='Q_v2')
SCORES = dict(Q_v1=('v1', 'total'), Q_vis=('keypoint', 'total'), Q_mot=('v1', 'residual_nuisance'),
              Q_v2=('keypoint', 'residual_nuisance'))
SIGMAS = (0.01, 0.02, 0.05)
SEEDS = 3
THETAS = (0.3, 0.4, 0.5, 0.6, 0.7)
PERTURBED = ('D0', 'D3', 'D3L', 'D5')
ORIGINAL_SCORE = sel.score_column


def repeatability(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x[1:]) & np.isfinite(x[:-1])
    var = np.nanvar(x)
    if ok.sum() < 10 or not np.isfinite(var) or var <= 1e-12:
        return np.nan
    return 1 - np.mean((x[1:][ok] - x[:-1][ok]) ** 2) / (2 * var)


def run(cand, frames, flags, fps, noise=None):
    if noise is None:
        sel.score_column = ORIGINAL_SCORE
    else:
        sigma, rng = noise
        sel.score_column = lambda t, v, m: ORIGINAL_SCORE(t, v, m) + rng.normal(0, sigma, len(t))
    try:
        geometry, _, _, _ = sel.select_video(cand, frames, flags, fps, 3, 1., None)
    finally:
        sel.score_column = ORIGINAL_SCORE
    sets = [frozenset(r[0] for r in geometry.get(f) or []) for f in frames]
    top = [(geometry.get(f) or [(None,)])[0][0] for f in frames]
    return sets, top


def agreement(a, b, duration):
    sa, ta = a
    sb, tb = b
    both = [i for i in range(len(sa)) if sa[i] or sb[i]]
    jac = np.mean([len(sa[i] & sb[i]) / len(sa[i] | sb[i]) for i in both]) if both else np.nan
    top = np.mean([ta[i] == tb[i] for i in both]) if both else np.nan
    sw = sum(1 for i in range(1, len(tb)) if tb[i] and tb[i - 1] and tb[i] != tb[i - 1]) / duration
    return jac, 100 * top, sw


def job(args):
    video, split, duration, cand, frames0, configs, kp, hom = args
    cand = cand.copy()
    for name, (v, m) in SCORES.items():
        cand[name] = ORIGINAL_SCORE(cand, v, m)
    avail = cand[cand.roi_status == 'AVAILABLE']
    frames = sorted(cand.frame_index.unique())
    fps = len(sorted(set(frames0) | set(frames))) / duration
    out = dict(e1=[], rank=[], e2=[], e3=[], e4=[])
    # E1 term repeatability within ROI over consecutive sampled frames
    index = {f: i for i, f in enumerate(frames)}
    for label, col in TERMS.items():
        rs = []
        for roi, rows in avail.groupby('roi_name'):
            series = np.full(len(frames), np.nan)
            series[[index[f] for f in rows.frame_index]] = rows[col].to_numpy(float)
            rs.append(repeatability(series))
        out['e1'].append(dict(video_id=video, split=split, term=label, repeatability=np.nanmedian(rs) if np.isfinite(rs).any() else np.nan))
    for name in SCORES:
        wide = avail.pivot_table(index='frame_index', columns='roi_name', values=name).reindex(frames)
        taus, margins = [], []
        for i in range(1, len(frames)):
            a, b = wide.iloc[i - 1], wide.iloc[i]
            common = a.notna() & b.notna()
            if common.sum() >= 4:
                taus.append(stats.kendalltau(a[common], b[common]).statistic)
            row = np.sort(b.dropna().to_numpy())[::-1]
            if len(row) >= 2:
                margins.append(row[0] - row[1])
        out['rank'].append(dict(video_id=video, split=split, score=name, kendall_tau=np.nanmean(taus) if taus else np.nan,
                                top_margin=np.nanmedian(margins) if margins else np.nan))
    # E2 score perturbation
    for cid in PERTURBED:
        ref = run(cand, frames, configs[cid], fps)
        for sigma in SIGMAS:
            for seed in range(SEEDS):
                rng = np.random.default_rng([seed, int(1000 * sigma), int(video[-3:])])
                j, t, s = agreement(ref, run(cand, frames, configs[cid], fps, (sigma, rng)), duration)
                out['e2'].append(dict(video_id=video, split=split, config=cid, sigma=sigma, seed=seed, jaccard=j,
                                      top_agreement=t, switches_per_sec=s))
    # E3 V' threshold sensitivity
    if kp is not None:
        ref = run(cand, frames, configs['D5'], fps)
        merged = cand.merge(kp, on=['frame_index', 'roi_name'], how='left')
        for theta in THETAS:
            c = merged.copy()
            c['visibility_keypoint'] = c[f'vp_{theta}']
            ok = c.visibility_keypoint.notna() & c[f'vp_0.5'].notna()
            rho = stats.spearmanr(c.loc[ok, 'visibility_keypoint'], c.loc[ok, 'vp_0.5']).statistic if ok.sum() > 10 else np.nan
            j, t, s = agreement(ref, run(c, frames, configs['D5'], fps), duration)
            out['e3'].append(dict(video_id=video, split=split, theta=theta, spearman_vs_0_5=rho, jaccard=j,
                                  top_agreement=t, switches_per_sec=s))
    # E4 motion estimator sensitivity
    if hom is not None:
        ref = run(cand, frames, configs['D6'], fps)
        c = cand.drop(columns=['motion_nuisance']).merge(hom, on=['frame_index', 'roi_name'], how='left')
        ok = c.motion_nuisance.notna() & cand.motion_nuisance.notna().to_numpy()
        rho = stats.spearmanr(c.loc[ok, 'motion_nuisance'], cand.loc[ok.to_numpy(), 'motion_nuisance']).statistic if ok.sum() > 10 else np.nan
        j, t, s = agreement(ref, run(c, frames, configs['D6'], fps), duration)
        out['e4'].append(dict(video_id=video, split=split, spearman_Mn=rho, jaccard=j, top_agreement=t, switches_per_sec=s,
                              homography_share=float(hom.global_status.eq('HOMOGRAPHY').mean()) if 'global_status' in hom else np.nan))
    return out


def keypoint_visibility_table(out, video, cand, pose, names):
    """V' at every threshold in THETAS for the AVAILABLE ROI-frames of one video, from the stored keypoints."""
    mapper = AnatomicalROIMapper(names, ev.CFG['phase1']['keypoint_confidence'])
    pose = pose.set_index('frame_index')
    rows = []
    for f, part in cand[cand.roi_status == 'AVAILABLE'].groupby('frame_index'):
        if f not in pose.index or not isinstance(pose.loc[f, 'keypoints_path'], str):
            continue
        points = terms.keypoint_table(json.loads((out/'pose_sam2'/pose.loc[f, 'keypoints_path']).read_text()).get('keypoints') or [])
        for roi in part.roi_name:
            rows.append(dict(frame_index=f, roi_name=roi, **{f'vp_{t}': terms.keypoint_visibility(mapper, roi, points, t)
                                                            for t in THETAS}))
    return pd.DataFrame(rows)


def kp_job(args):
    return keypoint_visibility_table(*args)


def summarize(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return 'n/a'
    _, lo, hi = bootstrap_ci(v)
    return f'{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f} [{lo:.3f}, {hi:.3f}]'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--max-videos', type=int, default=0)
    a = ap.parse_args(argv)
    run_dir = (ROOT/a.run).resolve()
    work, out = V2/'runs'/run_dir.name, run_dir/'outputs'
    target = V2/'reports'/run_dir.name/'consistency'
    target.mkdir(parents=True, exist_ok=True)
    configs = ev.configurations(json.loads((work/'tuning_log.json').read_text())['chosen'])
    manifest = pd.read_csv(run_dir/'manifest.csv')
    if a.max_videos:
        manifest = manifest.head(a.max_videos)
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False, usecols=ev.CAND_COLUMNS)
    feat = pd.read_csv(work/'features/roi_features_median.csv')
    cand = cand.merge(feat[['video_id', 'frame_index', 'roi_name', 'visibility_keypoint', 'motion_nuisance',
                            'motion_total', 'motion_residual', 'respiratory_ratio']],
                      on=['video_id', 'frame_index', 'roi_name'], how='left')
    hom_path = work/'features/roi_features_homography.csv'
    hom = pd.read_csv(hom_path) if hom_path.is_file() else None
    frames0 = pd.read_csv(out/'confidence/roi_confidence.csv', usecols=['video_id', 'frame_index'], low_memory=False)
    names = json.loads((out/'pose_sam2/experiment_metadata.json').read_text())['keypoint_names']
    pose = pd.read_csv(out/'pose_sam2/pose_results.csv', low_memory=False, usecols=['video_id', 'frame_index', 'keypoints_path'])
    started = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as pool:
        kps = list(pool.map(kp_job, [(out, r.video_id, cand[cand.video_id == r.video_id], pose[pose.video_id == r.video_id], names)
                                     for r in manifest.itertuples()]))
        jobs = []
        for r, kp in zip(manifest.itertuples(), kps):
            h = None
            if hom is not None:
                h = hom[hom.video_id == r.video_id][['frame_index', 'roi_name', 'motion_nuisance', 'global_status']] \
                    if 'motion_nuisance' in hom else None
            jobs.append((r.video_id, r.split, float(r.duration_sec), cand[cand.video_id == r.video_id],
                         frames0[frames0.video_id == r.video_id].frame_index.tolist(), configs, kp, h))
        results = list(pool.map(job, jobs))
    tables = {k: pd.DataFrame([row for r in results for row in r[k]]) for k in ('e1', 'rank', 'e2', 'e3', 'e4')}
    for k, t in tables.items():
        if len(t):
            t.to_csv(target/f'{k}_per_video.csv', index=False)

    # summaries and paired tests (video = unit)
    e1 = tables['e1'].pivot_table(index='video_id', columns='term', values='repeatability')
    rank = tables['rank'].pivot_table(index='video_id', columns='score', values=['kendall_tau', 'top_margin'])
    rows = [dict(item=f'repeatability {t}', value=summarize(e1[t])) for t in e1.columns]
    rows += [dict(item=f'{m} {s}', value=summarize(rank[(m, s)])) for m in ('kendall_tau', 'top_margin') for s in SCORES]
    pd.DataFrame(rows).to_csv(target/'table_e1_repeatability.csv', index=False)
    tests = []
    for a_, b_ in (('V', 'V_prime'), ('M', 'M_n'), ('Q', 'Q_prime')):
        tests.append(dict(family='repeatability', base=a_, candidate=b_, **paired_test(e1[a_], e1[b_])))
    for s in ('Q_vis', 'Q_mot', 'Q_v2'):
        for m in ('kendall_tau', 'top_margin'):
            tests.append(dict(family=m, base='Q_v1', candidate=s, **paired_test(rank[(m, 'Q_v1')], rank[(m, s)])))
    e2 = tables['e2'].groupby(['video_id', 'config', 'sigma'])[['jaccard', 'top_agreement', 'switches_per_sec']].mean().reset_index()
    rows2 = []
    for sigma in SIGMAS:
        part = e2[e2.sigma == sigma].pivot_table(index='video_id', columns='config', values=['jaccard', 'top_agreement', 'switches_per_sec'])
        for cid in PERTURBED:
            rows2.append(dict(sigma=sigma, config=cid, jaccard=summarize(part[('jaccard', cid)]),
                              top_agreement=summarize(part[('top_agreement', cid)]),
                              switches_per_sec=summarize(part[('switches_per_sec', cid)])))
        for cid in ('D3', 'D3L', 'D5'):
            for m in ('jaccard', 'top_agreement'):
                tests.append(dict(family=f'perturbation sigma={sigma} {m}', base='D0', candidate=cid,
                                  **paired_test(part[(m, 'D0')], part[(m, cid)])))
    pd.DataFrame(rows2).to_csv(target/'table_e2_perturbation.csv', index=False)
    if len(tables['e3']):
        e3 = tables['e3']
        pd.DataFrame([dict(theta=t, spearman_vs_0_5=summarize(p.spearman_vs_0_5), jaccard=summarize(p.jaccard),
                           top_agreement=summarize(p.top_agreement), switches_per_sec=summarize(p.switches_per_sec))
                      for t, p in e3.groupby('theta')]).to_csv(target/'table_e3_vprime_threshold.csv', index=False)
    if len(tables['e4']):
        e4 = tables['e4']
        pd.DataFrame([dict(spearman_Mn=summarize(e4.spearman_Mn), jaccard=summarize(e4.jaccard),
                           top_agreement=summarize(e4.top_agreement), switches_per_sec=summarize(e4.switches_per_sec),
                           homography_share=summarize(e4.homography_share))]).to_csv(target/'table_e4_motion_estimator.csv', index=False)
    tests = pd.DataFrame(tests)
    tests['p_holm'] = np.nan
    for fam, idx in tests.groupby('family').groups.items():
        tests.loc[idx, 'p_holm'] = holm(tests.loc[idx, 'p_value'].tolist())
    tests.to_csv(target/'paired_tests.csv', index=False)
    check = tables['e3'][tables['e3'].theta == .5] if len(tables['e3']) else pd.DataFrame()
    (target/'consistency_metadata.json').write_text(json.dumps(dict(
        sigmas=SIGMAS, seeds=SEEDS, thetas=THETAS, perturbed=PERTURBED, homography_available=hom is not None,
        regression_theta_0_5_jaccard=float(check.jaccard.min()) if len(check) else None,
        wall_time_sec=time.perf_counter() - started), indent=2))
    print(pd.read_csv(target/'table_e1_repeatability.csv').to_string(index=False))
    print(pd.read_csv(target/'table_e2_perturbation.csv').to_string(index=False))
    for f in ('table_e3_vprime_threshold.csv', 'table_e4_motion_estimator.csv'):
        if (target/f).is_file():
            print(pd.read_csv(target/f).to_string(index=False))
    print(tests[['family', 'base', 'candidate', 'n', 'mean_difference', 'p_value', 'p_holm']].round(4).to_string(index=False))
    print('theta=0.5 regression (min Jaccard vs D5, must be 1):', check.jaccard.min() if len(check) else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
