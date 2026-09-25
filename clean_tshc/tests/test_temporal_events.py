from tkh_abstraction.temporal import track, vi, transition


def hierarchy(year, groups):
    ids = sorted(v for g in groups for v in g)
    return dict(
        snapshot=year,
        partitions={str(l): groups for l in range(3)} | {"3": [[v] for v in ids]},
    )


def test_birth_death_and_primary_id_references():
    old = hierarchy(2020, [["a", "b"], ["c", "d"]])
    track(old, None)
    new = hierarchy(2022, [["a", "b"], ["e", "f"]])
    events = track(new, old)
    for level in range(3):
        rows = [e for e in events if e["level"] == level]
        assert {e["event_type"] for e in rows} == {"birth", "death", "continuation"}
        oldids = {r["persistent_id"] for r in old["supernodes"] if r["level"] == level}
        newids = {r["persistent_id"] for r in new["supernodes"] if r["level"] == level}
        assert all(
            set(e["old_ids"]) <= oldids and set(e["new_ids"]) <= newids for e in rows
        )
    assert transition(old, new)["0"]["lineage_retention"] == 1


def test_split_merge_ignore_one_node_contamination():
    old = hierarchy(2020, [["a", "b", "c", "d"], ["e", "f", "g", "h"]])
    track(old, None)
    balanced = hierarchy(2022, [["a", "b", "e", "f"], ["c", "d", "g", "h"]])
    events = track(balanced, old)
    assert sum(e["event_type"] == "merge" for e in events) == 6
    assert sum(e["event_type"] == "split" for e in events) == 6
    minor = hierarchy(2022, [["a", "b", "c", "e"], ["d", "f", "g", "h"]])
    events = track(minor, old)
    assert not any(e["event_type"] in ["merge", "split"] for e in events)
    assert len(minor["temporal_overlap"]["0"]) == 4


def test_vi_empty_singleton_and_label_invariance():
    assert vi([], []) == 0 and vi([1], [2]) == 0
    assert vi([0, 0, 1, 1], [5, 5, 7, 7]) == 0
    assert vi([0, 0, 1, 1], [0, 1, 0, 1]) > 0
