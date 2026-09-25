import pytest
import numpy as np
from tkh_abstraction.config import read_config

@pytest.fixture
def fixture_data():
    nodes=[]
    for i in range(12):
        y=2020 if i<8 else 2022
        nodes.append(dict(id=f'n{i}',type='method' if i%3 else 'claim',surface_form=('quantum atoms potential' if i%2 else 'graph neural learning')+f' topic {i}',year=1990 if i==10 else y,origin_year=1990 if i==10 else None,first_seen_year=y,last_seen_year=2022,provenance={}))
    members=[['n0','n2','n4','n6'],['n1','n3','n5','n7'],['n6','n7'],['n0','n8','n10'],['n1','n8','n9','n11']]
    edges=[dict(id=f'e{i}',relation_type='uses_technique',members=m,year=2020 if i<3 else 2022,provenance={'article_year':2020 if i<3 else 2022}) for i,m in enumerate(members)]
    return dict(meta={'synthetic':True,'date_semantics':{'node_first_seen_year':'earliest corpus mention','edge_year':'fact year','provenance_article_year':'asserting paper year'}},nodes=nodes,hyperedges=edges)

@pytest.fixture
def cfg():
    c=read_config(); c.update(budgets=[2,4],spectral_dim=3,neighbors=3,embedding_model='synthetic-hash',snapshots=[2020,2022]); return c
