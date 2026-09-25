"""One-command clean workflow; no imports from the previous project."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml
import numpy as np
import torch
from audit_data import audit
from tkh_abstraction.data import read, write, sha
from tkh_abstraction.embeddings import Encoder
from tkh_abstraction.hierarchy import build, semantic_links
from tkh_abstraction.temporal import track, transition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output")
    parser.add_argument(
        "--cache-dir",
        help="Optional separate embedding cache for a cold-vector reproduction",
    )
    parser.add_argument("--gate-d-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.cache_dir:
        cfg["cache_dir"] = args.cache_dir
    out = Path(args.output or cfg["output"])
    fresh_output = not out.exists() or not any(out.iterdir())
    out.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(
        list(Path("src").rglob("*.py"))
        + [
            Path("scripts/run_all.py"),
            Path("scripts/audit_data.py"),
            Path(args.config),
            Path("notes/target_annotations.json"),
            Path("notes/claim_annotations.json"),
        ]
    )
    source_hashes = {str(p): sha(p) for p in source_paths}
    write(out / "audit/source_manifest.json", source_hashes)
    torch.set_num_threads(cfg["threads"])
    started = time.perf_counter()
    graph, snaps, stats = audit(Path(cfg["data_dir"]), out, cfg["snapshots"])
    write(out / "audit/configuration.json", cfg)
    print("GATES A/B PASS", flush=True)
    enc = Encoder(
        cfg["construction_model"],
        cfg["construction_revision"],
        Path(cfg["cache_dir"]) / "construction",
        cfg["batch_size"],
    )
    vectors = enc.encode([f"{n['type']}: {n['surface_form']}" for n in graph["nodes"]])
    del enc
    base = {n["id"]: i for i, n in enumerate(graph["nodes"])}
    all_h = {}
    events = []
    transitions = {}
    for variant in (["temporal"] if args.gate_d_only else cfg["variants"]):
        hs = []
        prev = None
        transitions[variant] = {}
        for snap in snaps:
            x = vectors[[base[n["id"]] for n in snap["nodes"]]]
            print("BUILD", variant, snap["snapshot"], flush=True)
            h = build(snap, x, cfg, variant, prev)
            ee = track(h, prev, cfg["event_contribution"])
            events.extend(dict(e, variant=variant) for e in ee)
            if prev:
                transitions[variant][f"{prev['snapshot']}->{h['snapshot']}"] = (
                    transition(prev, h)
                )
            write(out / f"hierarchies/{variant}/hierarchy_{h['snapshot']}.json", h)
            hs.append(h)
            prev = h
            print("BUILT", variant, h["snapshot"], round(h["seconds"], 2), flush=True)
            if args.gate_d_only:
                write(
                    out / "audit/gate_D.json",
                    dict(
                        status="PASS",
                        variant=variant,
                        counts=[len(h["partitions"][str(i)]) for i in range(4)],
                        examples=h["partitions"]["0"][:5],
                    ),
                )
                return
        all_h[variant] = hs
    write(out / "temporal_events.json", events)
    # Remaining phases deliberately import evaluation only after construction.
    from tkh_abstraction.evaluation import execute_evaluation

    metrics = execute_evaluation(
        graph, snaps, stats, all_h, transitions, vectors, base, cfg, out
    )
    metrics["runtime_seconds"] = time.perf_counter() - started
    write(out / "metrics.json", metrics)
    from tkh_abstraction.plotting import deliver

    deliver(metrics, all_h, cfg, out)
    assert source_hashes == {
        str(p): sha(p) for p in source_paths
    }, "Source changed during this execution; rerun before accepting."
    write(
        out / "audit/execution_complete.json",
        dict(
            status="PASS",
            fresh_output_at_start=fresh_output,
            source_unchanged_during_run=True,
            metrics_sha256=sha(out / "metrics.json"),
            prediction_sha256=sha(out / "retrieval/predictions.json"),
        ),
    )
    print("WORKFLOW COMPLETE", round(metrics["runtime_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
