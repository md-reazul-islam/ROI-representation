from collections import defaultdict
import math


def per_video_metrics(rows):
    rows = sorted(rows, key=lambda r: r['frame_index'])
    total = len(rows)
    missed = sum(r['detector_status'] == 'MISSED' for r in rows)
    recovered = sum(r['recovered_status'] == 'ACCEPTED' for r in rows)
    run = longest = 0
    for row in rows:
        run = run + 1 if row['recovered_status'] == 'ACCEPTED' else 0
        longest = max(longest, run)
    return finish(dict(total_processed_frames=total, detector_missed_frames=missed,
                       recovered_frames=recovered, remaining_missed_frames=missed-recovered,
                       longest_recovered_sequence=longest))


def finish(result):
    total, missed = result['total_processed_frames'], result['detector_missed_frames']
    result.update(detector_miss_rate=missed/total if total else math.nan,
                  recovered_miss_rate=result['remaining_missed_frames']/total if total else math.nan,
                  temporal_recovery_rate=result['recovered_frames']/missed if missed else math.nan,
                  false_recovery_rate=None, false_recovery_rate_status='NOT_AVAILABLE',
                  false_recovery_rate_reason='No verified frame-level infant presence or bounding-box ground truth',
                  ground_truth_status='UNAVAILABLE', metric_interpretation='observation_coverage_not_detection_accuracy')
    return result


def summarize_recovery(rows):
    videos = defaultdict(list)
    for row in rows:
        videos[row['video_id']].append(row)
    per_video = [dict(scope='video', video_id=v, split=g[0]['split'], **per_video_metrics(g)) for v, g in videos.items()]
    output = list(per_video)
    groups = defaultdict(list)
    for metric in per_video:
        groups[metric['split']].append(metric)
    groups['all'] = per_video
    for split, metrics in groups.items():
        result = dict(scope='overall' if split == 'all' else 'split', split=split)
        for key in ('total_processed_frames', 'detector_missed_frames', 'recovered_frames', 'remaining_missed_frames'):
            result[key] = sum(m[key] for m in metrics)
        result['longest_recovered_sequence'] = max((m['longest_recovered_sequence'] for m in metrics), default=0)
        output.append(finish(result))
    return output
