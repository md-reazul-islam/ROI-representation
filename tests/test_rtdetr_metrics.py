import pytest
from module1.detection.rtdetr_metrics import aggregate_metrics, verify_frame_alignment


def row(video, frame, status):
    return dict(video_id=video, frame_index=frame, timestamp_sec=frame / 20,
                detection_status=status, selected_detection=status == 'DETECTED',
                confidence=0.8 if status == 'DETECTED' else None)


def test_gaps_do_not_interleave_or_join_videos():
    rows = [row('a', 0, 'MISSED'), row('a', 1, 'MISSED'),
            row('b', 0, 'DETECTED'), row('b', 1, 'MISSED')]
    result = aggregate_metrics(rows)[0]
    assert result['number_of_detection_gaps'] == 2
    assert result['longest_consecutive_miss'] == 2
    assert result['average_gap_length'] == 1.5
    assert result['miss_rate'] == 0.75
    assert result['mean_confidence'] == 0.8


def test_alignment_ignores_candidate_count_and_order():
    a = row('a', 0, 'MISSED')
    b = row('a', 1, 'DETECTED')
    verify_frame_alignment([a, b, b], [b, a])


@pytest.mark.parametrize('change', ['missing', 'timestamp', 'video'])
def test_alignment_rejects_different_frames(change):
    expected = [row('a', 0, 'MISSED')]
    actual = [dict(expected[0])]
    if change == 'missing':
        actual = []
    elif change == 'timestamp':
        actual[0]['timestamp_sec'] = 1
    else:
        actual[0]['video_id'] = 'b'
    with pytest.raises(ValueError, match='do not match'):
        verify_frame_alignment(actual, expected)
