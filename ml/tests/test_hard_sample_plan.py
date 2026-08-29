from collections import Counter

from crrc_vision.hard_sample_plan import build_h1a_jobs


def _references() -> list[dict[str, object]]:
    return [
        {
            "reference_id": f"ref-{index:02d}",
            "source_scene_id": f"scene-{index:02d}",
            "source_split": "train",
            "source_reference_sha256": f"{index:064X}",
            "crop_path": f"crops/ref-{index:02d}.png",
        }
        for index in range(1, 13)
    ]


def test_h1a_jobs_have_frozen_quota_and_lineage() -> None:
    references = _references()
    topology = {row["reference_id"]: "nut_plate" for row in references}
    jobs = build_h1a_jobs(references, topology, seed=20260829)
    assert len(jobs) == 24
    assert Counter(job["intent"] for job in jobs) == {
        "ALIGNED": 4,
        "SUBTLE_DISPLACED": 6,
        "OBVIOUS_DISPLACED": 4,
        "DAMAGED_MARK": 4,
        "INSUFFICIENT": 3,
        "LOOKALIKE": 3,
    }
    assert all(job["source_split"] == job["eligible_split"] == "train" for job in jobs)
    by_reference: dict[str, set[str]] = {}
    for job in jobs:
        by_reference.setdefault(str(job["reference_id"]), set()).add(str(job["intent"]))
    assert all(len(values) == 2 for values in by_reference.values())


def test_h1a_jobs_are_deterministic_and_hash_bound() -> None:
    references = _references()
    topology = {row["reference_id"]: "double_nut" for row in references}
    first = build_h1a_jobs(references, topology, seed=20260829)
    second = build_h1a_jobs(references, topology, seed=20260829)
    assert first == second
    assert all(len(str(job["prompt_sha256"])) == 64 for job in first)
