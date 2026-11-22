from module1_v2 import selection as sel


def test_two_point_box_becomes_rectangle_only_in_v2_mode():
    box = [[10, 40], [30, 20]]                         # corners in any order
    assert sel.pixel_polygon(box, 'skip') is None    # v1: no pixels
    assert sel.pixel_polygon(box, 'rectangle') == [[10, 20], [30, 20], [30, 40], [10, 40]]


def test_polygons_unchanged_and_degenerate_boxes_dropped():
    poly = [[0, 0], [5, 0], [5, 5]]
    assert sel.pixel_polygon(poly, 'skip') is poly
    assert sel.pixel_polygon(poly, 'rectangle') is poly
    assert sel.pixel_polygon([[3, 3], [3, 9]], 'rectangle') is None   # zero width
    assert sel.pixel_polygon(None, 'rectangle') is None
    assert sel.pixel_polygon([[1, 1]], 'rectangle') is None


def test_v2_configurations_carry_the_flag_and_d0_stays_v1():
    import importlib.util, json, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('ev', root / 'scripts/run_v2_evaluation.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    chosen = json.loads((root / 'runs/final34_paper/tuning_log.json').read_text())['chosen']
    c = ev.configurations(chosen)
    assert c['D0'].get('box_roi_pixels', 'skip') == 'skip'
    assert all(c[k]['box_roi_pixels'] == 'rectangle' for k in ('D1', 'D2', 'D3', 'D3L', 'D4', 'D5', 'D6', 'D7', 'I-B'))
    assert all(c[k].get('box_roi_pixels', 'skip') == 'skip' for k in ('I-V', 'I-M', 'I-S'))
