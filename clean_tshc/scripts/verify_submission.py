"""Audit real deliverables; compare a fresh complete run without deleting outputs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from tkh_abstraction.data import read, write, sha, load_graph
from tkh_abstraction.hierarchy import validate


def scientific(x):
    if isinstance(x, dict):
        return {
            k: scientific(v)
            for k, v in x.items()
            if k
            not in ["seconds", "runtime_seconds", "construction_runtimes", "cache_dir"]
        }
    if isinstance(x, list):
        return [scientific(v) for v in x]
    return x


def common_equal(new, old):
    """Permit only added audit fields; every earlier scientific field must agree."""
    if isinstance(old, dict):
        return all(
            k in new and common_equal(new[k], v)
            for k, v in old.items()
            if k
            not in ["seconds", "runtime_seconds", "construction_runtimes", "cache_dir"]
        )
    if isinstance(old, list):
        return len(new) == len(old) and all(
            common_equal(a, b) for a, b in zip(new, old)
        )
    return new == old


def audit(out, reference=None):
    m = read(out / "metrics.json")
    suites = ET.parse('notes/tests.xml').getroot().findall('testsuite')
    assert suites and all(int(s.get(key, 0)) == 0 for s in suites for key in ['errors', 'failures', 'skipped'])
    passed_tests = sum(int(s.get('tests', 0)) for s in suites)
    assert passed_tests >= 24
    cfg = m["config"]
    graph = load_graph(Path(cfg["data_dir"]))
    nodes = {n["id"]: n for n in graph["nodes"]}
    counts = {}
    hlookup = {}
    for v in cfg["variants"]:
        for year in cfg["snapshots"]:
            h = read(out / f"hierarchies/{v}/hierarchy_{year}.json")
            snap = read(out / f"snapshots/snapshot_{year}.json")
            ids = {n["id"] for n in snap["nodes"]}
            validate(h, ids, cfg["budgets"])
            hlookup[v, year] = h
            assert all(
                e["year"] <= year
                and (e["provenance"].get("article_year") or e["year"]) <= year
                and set(e["members"]) <= ids
                for e in snap["hyperedges"]
            )
            recs = {r["id"]: r for r in h["supernodes"]}
            for l in range(4):
                rr = [r for r in recs.values() if r["level"] == l]
                assert len({r["persistent_id"] for r in rr}) == len(rr)
                for r in rr:
                    assert r["size"] == len(r["member_ids"])
                    if l:
                        assert r["id"] in recs[r["parent_id"]]["child_ids"]
                    if l < 3:
                        assert set(r["member_ids"]) == {
                            v for c in r["child_ids"] for v in recs[c]["member_ids"]
                        }
                        if l < 2:
                            assert (
                                r["label"]
                                and r["gloss"]
                                and (out / f"labels/inputs/{v}/{r['id']}.json").exists()
                            )
            for l, ee in h["coarsened_hyperedges"].items():
                eligible = {r["id"] for r in recs.values() if r["level"] == int(l)}
                assert len(ee) == len(snap["hyperedges"])
                original = {e["id"]: e for e in snap["hyperedges"]}
                for e in ee:
                    assert (
                        set(e["supernodes"]) <= eligible
                        and sum(e["multiplicity"].values())
                        == len(original[e["id"]]["members"])
                        == e["original_arity"]
                    )
                    assert (
                        e["internal"] == (len(e["supernodes"]) == 1)
                        and e["provenance"] == original[e["id"]]["provenance"]
                    )
            counts[f"{v}/{year}"] = [len(h["partitions"][str(l)]) for l in range(4)]
    events = read(out / "temporal_events.json")
    for e in events:
        old = hlookup.get((e["variant"], e["from_snapshot"]))
        new = hlookup[e["variant"], e["to_snapshot"]]
        oldids = (
            {r["persistent_id"] for r in old["supernodes"] if r["level"] == e["level"]}
            if old
            else set()
        )
        newids = {
            r["persistent_id"] for r in new["supernodes"] if r["level"] == e["level"]
        }
        assert set(e["old_ids"]) <= oldids and set(e["new_ids"]) <= newids
    labels = read(out / "labels/faithfulness.json")
    for p in labels:
        gen = {r["id"] for r in p["generation_input"]["generation_members"]}
        held = set(p["held_out_ids"])
        assert not gen & held and all(
            nodes[v]["first_seen_year"] <= p["snapshot"] for v in gen | held
        )
        assert (
            read(out / f"labels/inputs/{p['variant']}/{p['id']}.json")
            == p["generation_input"]
        )
        if p["nli"]:
            pp = p["nli"]["probabilities"]
            assert abs(sum(pp.values()) - 1) < 1e-5
            assert p["nli"]["label"] == max(pp, key=pp.get) and p[
                "overclaim_proxy"
            ] == (p["nli"]["label"] != "entailment")
        else:
            assert p["overclaim_proxy"] is None and not p["evaluable"]
    for v in cfg["variants"]:
        assert len(m["perturbation_stability"][v]["rows"]) == 5
        for r in m["perturbation_stability"][v]["levels"].values():
            assert r["n"] == 5 and len(r["ci95"]) == 2
        for levels in m["coherence"][v].values():
            for r in levels.values():
                assert r["null"]["n"] >= 100 and r["null_macro"]["n"] >= 100
    pred = read(out / "retrieval/predictions.json")
    assert len(pred) == 90
    assert (
        sha(out / "retrieval/predictions.json")
        == read(out / "audit/gate_J.json")["sha256"]
    )
    for p in pred:
        assert (
            p["comparisons"]
            == p["leaf_comparisons"] + p["cluster_comparisons"]
            == len(p["trace"])
        )
        flat = next(
            x
            for x in pred
            if x["question_id"] == p["question_id"] and x["system"] == "flat"
        )
        scores = {r["node_id"]: r["score"] for r in flat["ranked"]}
        assert all(r["score"] == scores[r["node_id"]] for r in p["ranked"])
    support = read(out / "audit/benchmark_support_audit.json")
    assert len(support["records"]) == support["target_occurrences"]
    assert set(r["status"] for r in support["records"]) <= set(
        ["EXACT", "ALIAS", "COMPOSITE", "PARTIAL", "ABSENT"]
    )
    assert all(set(r["node_ids"]) <= nodes.keys() for r in support["records"])
    for filename in [
        "snapshot_growth",
        "arity_distribution",
        "coherence_vs_null",
        "temporal_stability",
        "perturbation_stability",
        "retrieval_work_tradeoff",
    ]:
        assert (out / f"figures/{filename}.png").stat().st_size > 1000
    report = (out / "report.md").read_text(encoding="utf8")
    assert "@AUDIT@" not in report and "P1 Laminarity" in report
    result = dict(
        status="PASS",
        hierarchy_counts=counts,
        label_records=len(labels),
        event_counts=dict(Counter(e["event_type"] for e in events)),
        predictions=len(pred),
        metrics_sha256=sha(out / "metrics.json"),
        report_words=len(report.split()),
        tests=dict(passed=passed_tests, failures=0, errors=0, skipped=0, source='notes/tests.xml'),
    )
    write(out / "audit/acceptance.json", result)
    if reference:
        previous = read(reference / "metrics.json")
        assert scientific(m)==scientific(previous), "Scientific metrics differ: inspect before accepting reproduction."
        source_now=read(out/'audit/source_manifest.json');source_before=read(reference/'audit/source_manifest.json')
        changed_sources=[k for k in source_now if source_now[k]!=source_before.get(k)]
        recovered=reference/'audit/recovered_execution.json'
        if changed_sources:
            allowed={'src/tkh_abstraction/benchmark.py','src/tkh_abstraction/evaluation.py','src/tkh_abstraction/plotting.py','scripts/run_all.py','notes/claim_annotations.json'}
            assert recovered.exists() and {k.replace('\\','/') for k in changed_sources}<=allowed,changed_sources
        assert read(out/'audit/input_manifest.json')==read(reference/'audit/input_manifest.json')
        for (v, year), h in hlookup.items():
            before = read(reference / f"hierarchies/{v}/hierarchy_{year}.json")
            assert (
                h["partitions"] == before["partitions"]
                and h["supernodes"] == before["supernodes"]
            )
        assert labels == read(reference / "labels/faithfulness.json")
        assert pred == read(reference / "retrieval/predictions.json")
        record=read(out/'audit/execution_complete.json')
        assert record['source_unchanged_during_run'] and record['fresh_output_at_start']
        assert (reference/'audit/execution_complete.json').exists() or recovered.exists()
        result["K"] = dict(
            status="PASS",
            fresh_directory=str(out),
            reference_directory=str(reference),
            all_scientific_fields_equal=True,
            input_manifests_equal=True,
            changed_sources=changed_sources,
            reference_evaluation_recovered=recovered.exists(),
            separate_embedding_cache=m['config']['cache_dir']!=previous['config']['cache_dir'],
            memberships_equal=True,
            labels_and_probabilities_equal=True,
            predictions_equal=True,
            reference_metrics_sha256=sha(reference / "metrics.json"),
            fresh_metrics_sha256=sha(out / "metrics.json"),
            scope="Ordinary complete fresh-output workflow with fixed final source and exact original inputs, compared against independently executed hierarchy/NLI/perturbation/sensitivity/prediction artifacts whose evaluation was recovered after a null-DOI bug and completed with one inspected claim annotation. Changed source is restricted to evaluation/reporting and its manifest; construction, scoring, labels, parameters and representations are unchanged. Runtime/cache path excluded from scientific equality. Model and content-addressed embedding caches may be reused. Earlier separate-cache vector regeneration is additional evidence. No historical project results used.",
        )
        write(out / "audit/gate_K.json", result["K"])
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--compare")
    args = ap.parse_args()
    print(
        json.dumps(
            audit(Path(args.output), Path(args.compare) if args.compare else None),
            indent=2,
        )
    )
