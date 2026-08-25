from crrc_vision.selection import SelectionCandidate, select_representatives


def candidate(
    path: str,
    group: str,
    split: str,
    focus: float,
    count: int,
) -> SelectionCandidate:
    return SelectionCandidate(path, group, split, focus, count)


def test_selection_is_deterministic_and_unique_by_group():
    rows = [
        candidate("a1.jpg", "g1", "train", 10.0, 1),
        candidate("a2.jpg", "g1", "train", 20.0, 1),
        candidate("b.jpg", "g2", "train", 15.0, 8),
        candidate("c.jpg", "g3", "val", 12.0, 3),
        candidate("d.jpg", "g4", "val", 11.0, 0),
    ]

    first = select_representatives(rows, target=4, val_count=2)
    second = select_representatives(list(reversed(rows)), target=4, val_count=2)

    assert first == second
    assert len({item.scene_group for item in first}) == 4
    assert "a2.jpg" in {item.relative_path for item in first}
    assert sum(item.split == "val" for item in first) == 2


def test_selection_stratifies_candidate_density_before_focus():
    rows = [
        candidate("zero.jpg", "g1", "train", 1.0, 0),
        candidate("low.jpg", "g2", "train", 1.0, 1),
        candidate("medium.jpg", "g3", "train", 1.0, 4),
        candidate("high.jpg", "g4", "train", 1.0, 20),
        candidate("extra.jpg", "g5", "train", 100.0, 4),
    ]

    selected = select_representatives(rows, target=4, val_count=0)

    assert {item.candidate_count for item in selected} == {0, 1, 4, 20}


def test_selection_rejects_impossible_target():
    rows = [candidate("a.jpg", "g1", "train", 1.0, 0)]

    try:
        select_representatives(rows, target=2, val_count=1)
    except ValueError as error:
        assert "target" in str(error)
    else:
        raise AssertionError("expected ValueError")
