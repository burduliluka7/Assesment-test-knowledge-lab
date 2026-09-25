"""Reload and audit actual exports rather than trusting successful process exit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tkh_abstraction.io import read_json,write_json,digest
from tkh_abstraction.hierarchy import validate_hierarchy
from tkh_abstraction.snapshots import snapshot
out=Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/full')
env=read_json(out/'environment.json'); cfg=env['config']; data=read_json(Path(cfg['data_dir'])/'tkh_collection10.json')
metrics=read_json(out/'metrics.json'); checks=[]; n_hierarchies=0
for v in cfg['variants']:
    known={}; pids={}
    for y in cfg['snapshots']:
        h=read_json(out/'variants'/v/f'hierarchy_{y}.json'); s=snapshot(data,y)
        validate_hierarchy(h,{n['id'] for n in s['nodes']}); n_hierarchies+=1
        for k,level in enumerate(h['levels']):
            known[y,k]={c['id'] for c in level}
            ids=[c['persistent_id'] for c in level]; assert len(ids)==len(set(ids))
            for c in level:
                assert c['label'] and c['gloss']; assert c['base_node_count']==len(c['member_ids'])
        for k,edges in h['coarse_hyperedges'].items():
            for e in edges:
                assert sum(e['multiplicities'].values())==e['original_arity']
                assert e['support_arity']==len(e['multiplicities'])
                assert e['internal']==(e['support_arity']==1)
                assert set(e['multiplicities'])<=known[y,int(k)]
        logs=read_json(out/'variants'/v/f'label_inputs_{y}.json')
        visible={n['id'] for n in s['nodes']}
        assert all(n['first_seen_year']<=y and n['id'] in visible for r in logs for n in r['input'])
    for e in read_json(out/'variants'/v/'temporal_events.json'):
        assert set(e['target_ids'])<=known[e['target_snapshot'],e['level']]
        if e['source_snapshot'] is not None: assert set(e['source_ids'])<=known[e['source_snapshot'],e['level']]
    for path in (out/'variants'/v).glob('hierarchy_*_benchmark.json'):
        h=read_json(path); y=h['snapshot']; s=snapshot(data,y)
        validate_hierarchy(h,{n['id'] for n in s['nodes']}); n_hierarchies+=1
        for k,level in enumerate(h['levels']): known[y,k]={c['id'] for c in level}
        events=read_json(out/'variants'/v/f'events_{y}_benchmark.json')
        for e in events:
            assert set(e['target_ids'])<=known[y,e['level']]
            if e['source_snapshot'] is not None: assert set(e['source_ids'])<=known[e['source_snapshot'],e['level']]
    vm=metrics['variants'][v]; perturb=vm['perturbation']
    assert len(perturb['records'])==5
    assert [r['seed'] for r in perturb['records']]==cfg['perturbation_seeds']
    assert all(abs(r['fraction']-.1)<.002 for r in perturb['records'])
    for year,result in vm['extrinsic'].items():
        if isinstance(result,dict) and result.get('status')=='computed':
            assert result['evaluated_questions']==14
            for q in result['per_question']:
                for b,r in q['hierarchical'].items():
                    assert r['inspections']<=int(b)
                    assert r['supporting_evidence']['inspections']<=r['supporting_evidence']['budget']
    checks.append(v)
assert metrics['config_hash']==digest(cfg)
assert metrics['status']=='computed'
assert data.get('meta',{}).get('synthetic',False)==(metrics['data_kind']=='synthetic')
root=Path('src/tkh_abstraction')
write_json(out/'verification.json',dict(status='passed',hierarchies_reloaded=n_hierarchies,variants=checks,
    delivered_source_hash=digest({p.relative_to(root).as_posix():p.read_text(encoding='utf-8') for p in sorted(root.rglob('*.py'))}),
    checks=['coverage','laminarity','budgets','singleton leaves','persistent id uniqueness','multiplicity conservation','edge support ids','temporal event references','snapshot-only label inputs','five perturbations per variant','retrieval budgets','config and data-kind consistency']))
print(f'Passed: {n_hierarchies} hierarchies, {len(checks)} variants; JSON, events, label inputs and metric checks')
