"""Read-only audit and immutable manifest for the existing submission."""
from pathlib import Path
import sys, hashlib, zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tkh_abstraction.io import read_json,write_json,digest,read_csv
from tkh_abstraction.hierarchy import validate_hierarchy
from tkh_abstraction.snapshots import snapshot

root=Path('.'); out=root/'artifacts/baseline_v1'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
core=['config.py','schema.py','snapshots.py','hypergraph.py','collapse.py','spectral.py','semantics.py','coarsening.py','hierarchy.py','temporal.py','baselines.py']
archive=zipfile.ZipFile(out/'submission.zip'); original=read_json(root/'artifacts/full/environment.json')
manifest=__import__('json').loads(archive.read('MANIFEST.json'))['files']
sources={f'src/tkh_abstraction/{name}':sha(root/'src/tkh_abstraction'/name) for name in core}
assert all(manifest[p]==h for p,h in sources.items()),'Core differs from archived verified submission'
data=read_json(root/'data/data/tkh_collection10.json')
assert digest(data)==original['data_hash']
files={}; partitions={}
for path in sorted((root/'artifacts/full/variants').glob('*/hierarchy_*.json')):
    h=read_json(path); s=snapshot(data,h['snapshot']); validate_hierarchy(h,{n['id'] for n in s['nodes']})
    files[path.as_posix()]=sha(path)
    partitions[path.as_posix()]=digest([[(c['id'],c['parent_id'],c['persistent_id'],c['member_ids']) for c in level] for level in h['levels']])
for p in (root/'artifacts/full').glob('**/temporal_events.json'): files[p.as_posix()]=sha(p)
m=read_json(out/'metrics.json'); q=read_csv(root/'data/data/questions.csv')
summary={v:dict(mapping=x['extrinsic']['2025']['mapping_coverage'],retrieval=x['extrinsic']['2025']['summary']['500'],faithfulness=x['faithfulness']['nli_proxy']['by_snapshot_level']) for v,x in m['variants'].items()}
raw=[dict(cutoff=y,raw_edge_year=sum(e['year']<=y for e in data['hyperedges']),visible_edges=len(snapshot(data,y)['hyperedges'])) for y in original['config']['snapshots']]
record=dict(schema_version='baseline-audit-v1',core_sources=sources,artifact_sha256=files,partition_hashes=partitions,
    construction_config=original['config'],data_hash=digest(data),original_archive_sha256=sha(out/'submission.zip'),
    question_count=len(q),baseline_summary=summary,raw_vs_visible=raw,prior_reproduction_check=read_json(root/'artifacts/full/reproduction_check.json'))
target=out/'manifest.json'
if target.exists(): assert read_json(target)==record,'Existing baseline manifest differs; do not overwrite history'
else: write_json(target,record)
print(f'Baseline recorded: {len(partitions)} validated hierarchies; {len(q)} questions; core matches original archive')
print(raw)
