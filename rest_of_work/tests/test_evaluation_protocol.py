import csv
from copy import deepcopy
import numpy as np
import pytest
from tkh_abstraction.evaluation.faithfulness import human_ratings
from tkh_abstraction.evaluation.statistics import interval
from tkh_abstraction.evaluation.extrinsic import metrics
from tkh_abstraction.pipeline import load
from tkh_abstraction.io import write_json

def test_human_ratings_are_never_fabricated(tmp_path):
    assert human_ratings(tmp_path/'missing.csv',[])['status']=='missing_data'
    sample=[dict(label='atoms',gloss='Atomic methods.',snapshot=2020,level=0),dict(label='graphs',gloss='Graph methods.',snapshot=2020,level=1)]
    p=tmp_path/'ratings.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['audit_id','label','gloss','rating']); w.writeheader()
        for i,(s,rating) in enumerate(zip(sample,['unsupported','accurate'])):
            w.writerow(dict(audit_id=f'A{i:04}',label=s['label'],gloss=s['gloss'],rating=rating))
    r=human_ratings(p,sample)
    assert r['by_snapshot_level']['2020:L0']['overclaim_rate']==1
    assert r['by_snapshot_level']['2020:L1']['overclaim_rate']==0
    other=deepcopy(sample); other[0]['gloss']='Changed.'
    with pytest.raises(ValueError): human_ratings(p,other)

def test_unmapped_target_not_zero_recall():
    result=dict(returned_ids=[],ranked_ids=[],inspections=3)
    r=metrics(result,['unknown'],{'unknown':dict(node_ids=[])})
    assert r['recall_mapped'] is None and r['recall_all_expected_lower_bound']==0
    assert r['status']=='missing_data' and r['f1'] is None and r['hit'] is None
    assert r['first_target_rank'] is None and r['censored_no_target']

def test_approximate_interval_contains_mean():
    r=interval([.1,.2,.3,.4,.5]); assert r['ci95'][0]<r['mean']<r['ci95'][1]
    assert r['std']==pytest.approx(np.std([.1,.2,.3,.4,.5],ddof=1))

def test_synthetic_cannot_overwrite_real_metrics(fixture_data,tmp_path):
    write_json(tmp_path/'data'/'tkh_collection10.json',fixture_data)
    with pytest.raises(ValueError,match='Synthetic runs'):
        load(dict(data_dir=str(tmp_path/'data'),output=str(tmp_path/'full')))
