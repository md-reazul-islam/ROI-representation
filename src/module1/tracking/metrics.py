"""Observation metrics for the selected infant track, without fabricated identity labels."""
import math
from collections import defaultdict


def tracking_metrics(rows):
    rows = sorted(rows, key=lambda r: r['frame_index'])
    total = len(rows)
    observed = [r for r in rows if r['observed']]
    longest = run = missing = longest_missing = segments = transitions = changes = 0
    previous = last_observed = None
    for row in rows:
        current = row['track_id'] if row['observed'] else None
        if current is None:
            missing += 1
            longest_missing = max(longest_missing, missing)
            run = 0
        else:
            missing = 0
            if current == previous:
                run += 1
                transitions += 1
            else:
                segments += 1
                run = 1
            longest = max(longest, run)
            if last_observed is not None and current != last_observed:
                changes += 1
            last_observed = current
        previous = current
    return dict(total_processed_frames=total, observed_track_frames=len(observed),
                missing_track_frames=total - len(observed),
                track_coverage=len(observed) / total if total else math.nan,
                track_continuity=transitions / (total - 1) if total > 1 else math.nan,
                continuous_transitions=transitions, possible_transitions=max(0, total - 1),
                number_of_track_segments=segments, number_of_fragments=max(0, segments - 1),
                longest_track=longest, longest_track_age=max((r['age'] for r in observed), default=0),
                consecutive_missing_frames=missing, longest_consecutive_missing_frames=longest_missing,
                observed_primary_id_changes=changes, id_switches=None,
                id_switches_status='NOT_AVAILABLE',
                id_switches_reason='No frame-level ground-truth identity annotations',
                ground_truth_status='UNAVAILABLE', metric_interpretation='tracker_observation_not_identity_accuracy')


def summarize_tracking(rows):
    videos = defaultdict(list)
    for row in rows:
        videos[row['video_id']].append(row)
    per_video = [dict(scope='video', video_id=v, split=g[0]['split'], **tracking_metrics(g)) for v, g in videos.items()]
    output = list(per_video)
    groups = defaultdict(list)
    for metric in per_video:
        groups[metric['split']].append(metric)
    groups['all'] = per_video
    for split, metrics in groups.items():
        result = dict(scope='overall' if split == 'all' else 'split', split=split,
                      id_switches=None, id_switches_status='NOT_AVAILABLE',
                      id_switches_reason='No frame-level ground-truth identity annotations',
                      ground_truth_status='UNAVAILABLE', metric_interpretation='tracker_observation_not_identity_accuracy')
        for field in ('total_processed_frames', 'observed_track_frames', 'missing_track_frames',
                      'continuous_transitions', 'possible_transitions', 'number_of_track_segments',
                      'number_of_fragments', 'observed_primary_id_changes'):
            result[field] = sum(m[field] for m in metrics)
        for field in ('longest_track', 'longest_track_age', 'longest_consecutive_missing_frames'):
            result[field] = max((m[field] for m in metrics), default=0)
        result['track_coverage'] = result['observed_track_frames'] / result['total_processed_frames'] if result['total_processed_frames'] else math.nan
        result['track_continuity'] = result['continuous_transitions'] / result['possible_transitions'] if result['possible_transitions'] else math.nan
        output.append(result)
    return output
