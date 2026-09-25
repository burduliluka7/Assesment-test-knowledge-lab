"""Populate optional cache only from exact already-recorded model/premise/hypothesis triples."""
from pathlib import Path
from tkh_abstraction.io import read_json,write_json,digest
from tkh_abstraction.config import read_config
cfg=read_config('configs/evaluation_v2.yaml')
metadata=dict(model=cfg['nli_model'],revision=cfg['nli_revision'],threshold_unchanged=.5,max_length=512)
cache={}
for p in Path('artifacts/evaluation/labels').glob('*/faithfulness.json'):
    result=read_json(p); assert result['model']==metadata
    hypotheses={(r['cluster_id'],r['snapshot']):r['gloss'] for r in result['rows'] if r['version']=='improved'}
    for r in result['rows']:
        for chunk in r['chunks']: cache[digest([chunk['premise'],r['gloss']])]=chunk['nli']
    for r in result['controls']: cache[digest([r['premise'],hypotheses[(r['cluster_id'],r['snapshot'])]])]=r['nli']
for a in read_json('artifacts/evaluation/claim_alignment.json').values():
    assert a['nli_model']==metadata
    for r in a['items']:
        for c in r['candidates']:
            if 'nli' in c: cache[digest([c['surface_form'],r['ground_truth_claim']])]=c['nli']
write_json(Path('.cache/nli')/(digest(metadata)+'.json'),cache)
print('Exact recorded NLI pairs cached:',len(cache))
