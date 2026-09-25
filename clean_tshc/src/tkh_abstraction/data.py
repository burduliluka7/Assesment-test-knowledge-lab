import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf8",
    )


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_graph(folder):
    graph = read(Path(folder) / "tkh_collection10.json")
    nodes = graph["nodes"]
    edges = graph["hyperedges"]
    ids = {n["id"] for n in nodes}
    assert len(ids) == len(nodes) and len({e["id"] for e in edges}) == len(edges)
    for n in nodes:
        assert all(
            k in n
            for k in ["id", "type", "surface_form", "first_seen_year", "provenance"]
        )
        assert isinstance(n["first_seen_year"], int) and n["surface_form"].strip()
    for e in edges:
        assert all(
            k in e for k in ["id", "relation_type", "members", "year", "provenance"]
        )
        assert set(e["members"]) <= ids and len(set(e["members"])) == len(e["members"])
        assert len(e["members"]) > 1
    return graph


def manifest(folder):
    rows = []
    for p in sorted(Path(folder).iterdir()):
        if not p.is_file():
            continue
        summary = {}
        if p.suffix == ".json":
            x = read(p)
            summary = {
                "root_type": type(x).__name__,
                "top_level_keys": list(x),
                "record_counts": {
                    k: len(v) for k, v in x.items() if isinstance(v, list)
                },
                "metadata": x.get("meta"),
            }
        elif p.suffix == ".csv":
            with p.open(encoding="utf-8-sig", newline="") as f:
                dialect = csv.Sniffer().sniff(f.read(4096), delimiters=",;")
                f.seek(0)
                r = list(csv.DictReader(f, dialect=dialect))
            summary = {
                "records": len(r),
                "columns": list(r[0]) if r else [],
                "delimiter": dialect.delimiter,
            }
        rows.append(
            dict(filename=p.name, bytes=p.stat().st_size, sha256=sha(p), schema=summary)
        )
    return rows


def describe(snap, previous=None):
    nodes = snap["nodes"]
    edges = snap["hyperedges"]
    arities = [len(e["members"]) for e in edges]
    oldn = {n["id"] for n in previous["nodes"]} if previous else set()
    olde = {e["id"] for e in previous["hyperedges"]} if previous else set()
    return dict(
        snapshot=snap["snapshot"],
        nodes=len(nodes),
        hyperedges=len(edges),
        node_types=dict(Counter(n["type"] for n in nodes)),
        relation_types=dict(Counter(e["relation_type"] for e in edges)),
        arity=dict(
            min=min(arities),
            max=max(arities),
            mean=float(np.mean(arities)),
            median=float(np.median(arities)),
            histogram=dict(Counter(arities)),
            fraction_gt2=float(np.mean(np.array(arities) > 2)),
        ),
        first_seen_span=[
            min(n["first_seen_year"] for n in nodes),
            max(n["first_seen_year"] for n in nodes),
        ],
        new_nodes=len({n["id"] for n in nodes} - oldn),
        new_edges=len({e["id"] for e in edges} - olde),
        node_growth_percent=(
            100 * (len(nodes) - len(oldn)) / len(oldn) if oldn else None
        ),
        edge_growth_percent=(
            100 * (len(edges) - len(olde)) / len(olde) if olde else None
        ),
    )
