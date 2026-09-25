"""Write a clearly marked fixture plus explicitly prescribed split/merge partitions."""
import json
from pathlib import Path
nodes=[]
for i in range(12):
    year=2020 if i<8 else 2022
    nodes.append(dict(id=f'n{i}',type='method',surface_form=('quantum atomic potentials' if i%2 else 'graph neural learning')+f' topic {i}',
        year=1990 if i==10 else year,origin_year=1990 if i==10 else None,first_seen_year=year,last_seen_year=2022,provenance={}))
sets=[['n0','n2','n4','n6'],['n1','n3','n5','n7'],['n6','n7'],['n0','n8','n10'],['n1','n8','n9','n11']]
edges=[dict(id=f'e{i}',relation_type='uses_technique',members=m,year=2020 if i<3 else 2022,provenance={'article_year':2020 if i<3 else 2022}) for i,m in enumerate(sets)]
edges.append(dict(id='future_assertion',relation_type='claims',members=['n0','n10'],year=1999,provenance={'article_year':2022}))
fixture=dict(meta=dict(synthetic=True,date_semantics=dict(node_first_seen_year='earliest corpus mention',edge_year='fact year',provenance_article_year='asserting paper year')),nodes=nodes,hyperedges=edges)
out=Path('tests/fixtures/synthetic'); out.mkdir(parents=True,exist_ok=True)
(out/'tkh_collection10.json').write_text(json.dumps(fixture,indent=2),encoding='utf-8')
(out/'event_partitions.json').write_text(json.dumps(dict(description='Prescribed event cases, not an empirical clustering result',
    split=dict(previous=[['n0','n2','n4','n6']],current=[['n0','n2'],['n4','n6']]),
    merge=dict(previous=[['n1','n3'],['n5','n7']],current=[['n1','n3','n5','n7']])),indent=2),encoding='utf-8')
print(out)
