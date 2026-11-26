import pytest

from src.prelabel import (
    AMBIGUOUS,
    CLASS_DISAGREEMENT,
    OVERLAPPING,
    UNMATCHED,
    apply_class_fixes,
    match_predictions,
)
from src.validate_labels import Box


def box(class_id, x, y, w=0.1, h=0.1, line=0):
    return Box(class_id, x, y, w, h, line)


def categories(matches):
    return [category for _, category, _, _ in matches]


def test_one_original_box_matches_only_one_prediction():
    labeled = [box(1, 0.5, 0.5, line=1)]
    predicted = [box(1, 0.5, 0.5), box(1, 0.505, 0.5)]  # two near-identical predictions
    assert sorted(categories(match_predictions(predicted, labeled))) == [OVERLAPPING, UNMATCHED]


def test_best_iou_wins_in_greedy_matching():
    labeled = [box(1, 0.5, 0.5, line=1)]
    worse, better = box(1, 0.53, 0.5), box(1, 0.5, 0.5)
    matches = match_predictions([worse, better], labeled)
    assert categories(matches) == [UNMATCHED, OVERLAPPING]


def test_categories_by_iou_and_class():
    labeled = [box(1, 0.2, 0.2, line=1), box(7, 0.5, 0.5, line=2), box(1, 0.8, 0.8, line=3)]
    predicted = [
        box(1, 0.2, 0.2),  # same place, same class
        box(1, 0.5, 0.5),  # same place, original says police -> class disagreement
        box(1, 0.83, 0.8),  # shifted 0.03 of 0.1 -> IoU ~0.54
        box(1, 0.86, 0.8),  # partner already taken -> unmatched
        box(1, 0.1, 0.9),  # nothing there
    ]
    matches = match_predictions(predicted, labeled)
    assert categories(matches) == [
        OVERLAPPING,
        CLASS_DISAGREEMENT,
        OVERLAPPING,
        UNMATCHED,
        UNMATCHED,
    ]


def test_ambiguous_overlap():
    labeled = [box(1, 0.5, 0.5, line=1)]
    predicted = [box(1, 0.545, 0.5)]  # IoU = 0.055 / 0.145 ~ 0.38
    assert categories(match_predictions(predicted, labeled)) == [AMBIGUOUS]


def test_apply_class_fixes_changes_only_the_listed_line():
    text = "1 0.5 0.5 0.2 0.2\r\n11 0.1 0.1 0.05 0.05\r\n"
    fixes = {("a.png", 2): (11, 10)}
    assert apply_class_fixes(text, "a.png", fixes) == "1 0.5 0.5 0.2 0.2\n10 0.1 0.1 0.05 0.05\n"
    assert apply_class_fixes(text, "b.png", fixes) == "1 0.5 0.5 0.2 0.2\n11 0.1 0.1 0.05 0.05\n"


def test_apply_class_fixes_refuses_stale_fix():
    with pytest.raises(ValueError, match="expected class 11"):
        apply_class_fixes("1 0.5 0.5 0.2 0.2\n", "a.png", {("a.png", 1): (11, 10)})


def test_one_prediction_matches_only_one_original_box():
    # One wide prediction overlaps two original boxes (IoU 0.5 each): it takes one partner only.
    labeled = [box(1, 0.45, 0.5, line=1), box(1, 0.55, 0.5, line=2)]
    wide = box(1, 0.5, 0.5, w=0.2)
    matches = match_predictions([wide], labeled)
    assert len(matches) == 1
    _, _, original, iou = matches[0]
    assert original in labeled and iou == pytest.approx(0.5)


def test_diff_boxes_reports_each_kind_of_change():
    from src.annotation_changes import diff_boxes

    source = [
        box(1, 0.2, 0.2, line=1),  # unchanged
        box(1, 0.5, 0.5, line=2),  # adjusted
        box(1, 0.8, 0.2, line=3),  # reclassified
        box(10, 0.5, 0.9, line=4),  # shifted far away, same size -> moved
        box(4, 0.1, 0.9, line=5),  # false box -> removed
    ]
    final = [
        box(1, 0.2, 0.2),
        box(1, 0.51, 0.5),
        box(7, 0.8, 0.2),
        box(10, 0.5, 0.7),
        box(4, 0.9, 0.9, w=0.3, h=0.2),  # new object
    ]
    changes = sorted((change, s.line if s else None) for change, s, _ in diff_boxes(source, final))
    assert changes == [
        ("added", None),
        ("adjusted", 2),
        ("moved", 4),
        ("reclassified", 3),
        ("removed", 5),
    ]
