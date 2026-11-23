"""Tracking metrics on a hand-made example where every number is known in advance."""
import pytest

from safecross.vision.track_eval import calibration, evaluate


def _box(x):
    return (x, 0, x + 10, 30)


def _scene():
    # 10 frames, fps 10. GT person 1 (wheelchair) walks right; GT person 2 (pedestrian) stands still.
    gt = {f: [(1, _box(f * 2), "wheelchair"), (2, _box(100), "pedestrian")] for f in range(10)}
    pred = {}
    for f in range(10):
        P = []
        if f not in (3, 4, 5):                       # person 1 missed for 3 frames (0.3 s)...
            P.append((7 if f < 3 else 8, _box(f * 2), "wheelchair" if f != 8 else "pedestrian", 0.9))  # ...then new ID 8
        P.append((9, _box(100), "pedestrian", 0.9))  # person 2 always tracked, one ID
        if f == 6:
            P.append((50, _box(300), "pedestrian", 0.4))   # one phantom box
        if f == 2:
            P.append((60, _box(500), "pedestrian", 0.4))   # a box on an "ignore" person (not FP)
        pred[f] = P
    ignore = {2: [_box(500)]}
    return gt, pred, ignore


def test_clear_mot_and_idf1():
    gt, pred, ignore = _scene()
    s, tracks, bursts, changes = evaluate(gt, pred, [list(range(10))], fps=10.0, ignore=ignore)
    assert s["gt_boxes"] == 20 and s["FN"] == 3 and s["FP"] == 1 and s["IDSW"] == 1
    assert s["ignored_tracker_boxes"] == 1
    assert s["MOTA"] == pytest.approx(1 - 5 / 20)
    # IDF1: person 1 <-> ID 8 (4 frames, better than ID 7 with 3), person 2 <-> ID 9 (10); tracker boxes = 18
    assert s["IDTP"] == 14
    assert s["IDF1"] == pytest.approx(2 * 14 / (20 + 18))
    p1 = next(t for t in tracks if t["gt_id"] == 1)
    assert p1["detected_share"] == pytest.approx(0.7) and p1["id_changes"] == 1 and p1["group"] == "mobility_aid"
    assert p1["labelled_as_pedestrian_share"] == pytest.approx(1 / 7)
    assert [b["seconds"] for b in bursts] == [pytest.approx(0.3)] and bursts[0]["where"] == "middle"
    assert changes[0]["gap_seconds"] == pytest.approx(0.3)


def test_calibration_numbers():
    gt, pred, ignore = _scene()
    cal = calibration(*evaluate(gt, pred, [list(range(10))], fps=10.0, ignore=ignore))
    f = cal["failures_for_experiment_b"]
    assert f["recall"] == pytest.approx(0.7) and f["miss_burst_frames"] == pytest.approx(3.0)
    assert f["track_loss_prob"] == 1.0 and f["track_loss_duration_s"][0] == pytest.approx(0.3)
    assert f["false_positive_rate"] == pytest.approx(1.0)        # 1 phantom track in 1 s of video


def test_new_sequence_is_not_an_id_switch():
    gt = {f: [(1, _box(0), "pedestrian")] for f in range(6)}
    pred = {f: [(1 if f < 3 else 2, _box(0), "pedestrian", 0.9)] for f in range(6)}
    s, *_ = evaluate(gt, pred, [[0, 1, 2], [3, 4, 5]], fps=10.0)   # tracker restarted at frame 3
    assert s["IDSW"] == 0
