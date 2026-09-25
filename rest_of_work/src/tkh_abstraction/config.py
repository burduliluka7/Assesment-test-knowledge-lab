from copy import deepcopy
from pathlib import Path
import yaml

DEFAULT = dict(data_dir="data/data", output="artifacts/full", snapshots=[2020,2022,2024,2026],
    budgets=[12,40,120], seed=42, spectral_dim=8, neighbors=12,
    weights=dict(hyper=.30,spectral=.20,semantic=.35,temporal=.15), relation_weights={},
    embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    evaluation_model="BAAI/bge-small-en-v1.5", evaluation_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    nli_model="cross-encoder/nli-deberta-v3-small", nli_revision="fa2804872c3b4bd748f38c0185cc85775361e735", include_type=True,
    variants=["semantic","structural","static","temporal","pairwise"],
    perturbation_seeds=[11,23,37,53,71], null_permutations=50,
    retrieval_budgets=[50,100,250,500], retrieval_beam=3, match_threshold=.20,
    event_threshold=.10,nli_threshold=.50,faithfulness_sample_per_level=12,sensitivity=True)

def read_config(path=None):
    cfg=deepcopy(DEFAULT)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if not cfg['budgets'] or any(x<1 for x in cfg['budgets']) or sorted(cfg['budgets'])!=cfg['budgets']:
        raise ValueError("budgets must be positive, ordered coarse to fine")
    if sorted(set(cfg['snapshots']))!=cfg['snapshots']:
        raise ValueError("snapshots must be unique and increasing")
    if any(w<0 for w in cfg['weights'].values()) or sum(cfg['weights'].values())<=0:
        raise ValueError("objective weights must be nonnegative and nonzero")
    return cfg
