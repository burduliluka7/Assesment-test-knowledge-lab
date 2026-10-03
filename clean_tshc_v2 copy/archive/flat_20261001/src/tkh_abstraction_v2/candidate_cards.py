"""Capped native-hypergraph evidence; no transitive propagation or article claim dump."""
from collections import defaultdict
import re

FIELDS = ('direct_claims', 'task_problem', 'technical', 'dataset_metric', 'article', 'other')
TYPE_FIELD = {'claim': 'direct_claims', 'task': 'task_problem', 'problem': 'task_problem',
              'technique': 'technical', 'component': 'technical', 'dataset': 'dataset_metric',
              'metric': 'dataset_metric', 'article': 'article', 'method': 'other', 'cited_work': 'other'}


def eligible(node_type, answer_type, cfg):
    family = cfg['answer_families'].get(answer_type)
    return node_type != 'author' and (family is None or node_type in family)


def build_cards(snap, cfg):
    nodes = {n['id']: n for n in snap['nodes']}
    incident = defaultdict(list)
    for e in snap['hyperedges']:
        for ident in e['members']:
            incident[ident].append(e)
    claims = [n for n in snap['nodes'] if n['type'] == 'claim']
    articles = defaultdict(list)
    for n in snap['nodes']:
        if n['type'] == 'article':
            for aid in n.get('provenance', {}).get('articles', []):
                articles[aid].append(n)
    cards = {}
    for ident, n in sorted(nodes.items()):
        if n['type'] == 'author':
            continue
        pools = defaultdict(list)
        discarded = []
        for e in sorted(incident[ident], key=lambda x: x['id']):
            if e['relation_type'] not in cfg['content_relations']:
                continue
            if len(e['members']) > cfg['direct_arity_cap']:
                discarded.append(dict(hyperedge_id=e['id'], reason='high_arity_comembership', arity=len(e['members'])))
                continue
            for other in sorted(set(e['members']) - {ident}):
                node = nodes[other]
                field = TYPE_FIELD.get(node['type'])
                if not field or (field == 'article' and e['relation_type'] != 'presents'):
                    continue
                if field == 'direct_claims' and e['relation_type'] != 'claims':
                    continue
                pools[field].append(dict(node_id=other, hyperedge_ids=[e['id']], relation_type=e['relation_type'],
                    text=node['surface_form'], node_type=node['type'], first_seen_year=node['first_seen_year'],
                    provenance=node['provenance'], edge_provenance=e['provenance'], arity=len(e['members']),
                    support_kind='low_arity_incidence_not_directional_entailment', priority=0))
        # Conservative exact bounded name match. Short lower-case generic words are excluded.
        name = n['surface_form']
        mention_safe = len(name) >= 5 or (len(name) >= 3 and name.isupper())
        if mention_safe and n['type'] in {'method', 'technique', 'component', 'cited_work'}:
            pattern = re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)', re.I)
            for node in claims:
                if node['id'] != ident and pattern.search(node['surface_form']):
                    pools['direct_claims'].append(dict(node_id=node['id'], hyperedge_ids=[], relation_type='explicit_name_mention',
                        text=node['surface_form'], node_type=node['type'], first_seen_year=node['first_seen_year'],
                        provenance=node['provenance'], support_kind='literal_name_match_not_entailment', priority=1, arity=0))
        for aid in n.get('provenance', {}).get('articles', []):
            for node in articles[aid]:
                if node['id'] != ident:
                    pools['article'].append(dict(node_id=node['id'], hyperedge_ids=[], relation_type='shared_source_title',
                        text=node['surface_form'], node_type=node['type'], first_seen_year=node['first_seen_year'],
                        provenance=node['provenance'], support_kind='provenance_title_only', priority=2, arity=0))
        chosen = {}
        seen = {name.casefold()}
        blocks = [dict(field='identity', node_id=ident, text=f"NAME: {name}\nTYPE: {n['type']}")]
        for field in FIELDS:
            chosen[field] = []
            for item in sorted(pools[field], key=lambda x: (x['priority'], x['arity'], x['node_id'])):
                if item['text'].casefold() in seen:
                    continue
                seen.add(item['text'].casefold())
                if len(chosen[field]) >= cfg['field_caps'][field]:
                    discarded.append(dict(node_id=item['node_id'], reason='field_cap', field=field))
                    continue
                chosen[field].append(item)
                blocks.append(dict(field=field, node_id=item['node_id'], text=field.upper() + ': ' + item['text']))
        cards[ident] = dict(node_id=ident, name=name, node_type=n['type'], snapshot=snap['snapshot'],
            provenance=n['provenance'], fields=chosen, blocks=blocks, discarded=discarded,
            text='\n'.join(b['text'] for b in blocks))
    return cards
