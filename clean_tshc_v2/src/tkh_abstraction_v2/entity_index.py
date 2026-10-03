"""Gold-free canonical entities, separate evidence, and role-checked directed arcs."""
from collections import defaultdict
import hashlib
import re
import unicodedata

METHOD_TYPES = {'method', 'technique', 'component', 'cited_work'}
FIELDS = ('claims', 'explicit_mentions', 'tasks', 'problems', 'techniques', 'components', 'datasets', 'metrics', 'articles', 'graph_evidence')
FIELD = {'claim': 'claims', 'task': 'tasks', 'problem': 'problems', 'technique': 'techniques', 'component': 'components', 'dataset': 'datasets', 'metric': 'metrics', 'article': 'articles'}
TARGET_TYPES = {'addresses': {'task'}, 'solves': {'problem'}, 'uses_technique': {'technique'}, 'uses_component': {'component'}, 'evaluated_on': {'dataset', 'metric'}}


def normalize_name(value):
    value = unicodedata.normalize('NFKC', value).casefold()
    value = value.translate(str.maketrans({c: '-' for c in '\u2010\u2011\u2012\u2013\u2212'}))
    return ' '.join(value.split())


def safe_name(name):
    tokens = re.findall(r'\w+', name)
    identifier = bool(re.search(r'[a-z][A-Z]|\d', name) or (name.isupper() and len(name) >= 3))
    return len(name) >= 3 and (identifier or len(tokens) >= 2 and len(name) >= 10)


def name_matches(text, name):
    if not safe_name(name):
        return False
    return bool(re.search(r'(?<!\w)' + re.escape(normalize_name(name)) + r'(?!\w)', normalize_name(text)))


def group_entities(nodes, cfg, cutoff):
    buckets = defaultdict(list)
    for n in nodes:
        if n['type'] not in cfg['entity_types'] or n['first_seen_year'] > cutoff:
            continue
        family = 'approach' if n['type'] in METHOD_TYPES else n['type']
        buckets[(family, normalize_name(n['surface_form']))].append(n)
    entities, node_to_entity = {}, {}
    for _, members in sorted(buckets.items()):
        members = sorted(members, key=lambda n: n['id'])
        ids = [n['id'] for n in members]
        eid = 'entity:' + ids[0]
        names = sorted({n['surface_form'] for n in members}, key=lambda s: (len(s), s))
        # Alias annotations have no individual dates: suppress aliases on nodes
        # with later observations when constructing an earlier snapshot.
        aliases = sorted({a for n in members if n.get('last_seen_year', n['first_seen_year']) <= cutoff
                          for a in n.get('aliases', []) if isinstance(a, str) and a.strip()} - set(names))
        entities[eid] = dict(entity_id=eid, name=names[0], names=names, aliases=aliases,
            types=sorted({n['type'] for n in members}), node_ids=ids,
            first_seen_year=min(n['first_seen_year'] for n in members),
            provenance=[dict(node_id=n['id'], **n.get('provenance', {})) for n in members],
            evidence={f: [] for f in FIELDS})
        node_to_entity.update({n: eid for n in ids})
    return dict(sorted(entities.items())), node_to_entity


def allowed_entity(entity, intent, cfg):
    types = intent.get('answer_types') or ([intent['answer_type']] if intent.get('answer_type') else [])
    families = [cfg['answer_families'][t] for t in types if t in cfg['answer_families']]
    return not families or bool(set(entity['types']) & {t for f in families for t in f})


def build_index(snap, cfg):
    cutoff = snap['snapshot']
    nodes = {n['id']: n for n in snap['nodes'] if n['first_seen_year'] <= cutoff}
    entities, node_to_entity = group_entities(list(nodes.values()), cfg, cutoff)
    items = {i: dict(evidence_id=i, node_id=i, text=n['surface_form'], type=n['type'],
        first_seen_year=n['first_seen_year'], provenance=n.get('provenance', {}))
        for i, n in sorted(nodes.items()) if n['type'] in FIELD}
    links = defaultdict(dict)
    arcs, rejected = defaultdict(list), []
    articles_by_source = defaultdict(list)
    for n in nodes.values():
        if n['type'] == 'article':
            for aid in n.get('provenance', {}).get('articles', []):
                articles_by_source[aid].append(n['id'])

    def arc(source, dest, edge, kind):
        arcs[source].append(dict(source_node=source, destination_node=dest, hyperedge_id=edge['id'],
            relation=edge['relation_type'], year=edge['year'], provenance=edge.get('provenance', {}), role=kind))

    def attach(entity_node, evidence_node, field, kind, edges=(), candidate_specific=True):
        eid = node_to_entity.get(entity_node)
        if eid is None or evidence_node not in items or evidence_node in entities[eid]['node_ids']:
            return
        key = (field, evidence_node)
        row = links[eid].setdefault(key, dict(evidence_id=evidence_node, field=field,
            support_kind=kind, candidate_specific=candidate_specific, paths=[]))
        path = [dict(hyperedge_id=e['id'], relation=e['relation_type'], year=e['year'], provenance=e.get('provenance', {})) for e in edges]
        if path not in row['paths']:
            row['paths'].append(path)

    for edge in sorted(snap['hyperedges'], key=lambda e: e['id']):
        if edge['year'] > cutoff or (edge.get('provenance', {}).get('article_year') or edge['year']) > cutoff or not set(edge['members']) <= set(nodes):
            continue
        members = [nodes[i] for i in edge['members']]
        rel = edge['relation_type']
        by_type = defaultdict(list)
        for n in members:
            by_type[n['type']].append(n['id'])
        valid = False
        if rel in TARGET_TYPES:
            subjects = by_type['method']
            targets = [n['id'] for n in members if n['type'] in TARGET_TYPES[rel]]
            valid = len(subjects) == 1 and len(targets) + 1 == len(members)
            if valid:
                for target in targets:
                    attach(subjects[0], target, FIELD[nodes[target]['type']], 'validated_method_target', [edge])
                    arc(target, subjects[0], edge, 'typed_evidence_to_method')
        elif rel == 'claims':
            valid = len(by_type['article']) == 1 and len(by_type['claim']) == 1 and len(by_type['method']) <= 1 and len(members) == 2 + len(by_type['method'])
            if valid:
                claim, article = by_type['claim'][0], by_type['article'][0]
                arc(claim, article, edge, 'asserting_article')
                attach(article, claim, 'claims', 'article_asserts_claim', [edge])
                for method in by_type['method']:
                    attach(method, claim, 'claims', 'explicit_method_claim', [edge])
                    arc(claim, method, edge, 'explicit_method_claim')
        elif rel == 'presents':
            valid = len(members) == 2 and len(by_type['article']) == 1 and len(by_type['method']) == 1
            if valid:
                article, method = by_type['article'][0], by_type['method'][0]
                attach(method, article, 'articles', 'presenting_article', [edge], False)
                arc(article, method, edge, 'presents')
        elif rel == 'cites':
            valid = len(by_type['article']) == 1 and len(by_type['cited_work']) == len(members)-1
            if valid:
                for cited in by_type['cited_work']:
                    arc(by_type['article'][0], cited, edge, 'citation_context_not_entailment')
        if rel in set(TARGET_TYPES) | {'claims', 'presents', 'cites'} and not valid:
            rejected.append(dict(hyperedge_id=edge['id'], reason='role_schema_mismatch'))
    claim_items = [n for n in nodes.values() if n['type'] == 'claim']
    for eid, entity in entities.items():
        for nid in entity['node_ids']:
            n = nodes[nid]
            for aid in n.get('provenance', {}).get('articles', []):
                for article in sorted(articles_by_source.get(aid, [])):
                    attach(nid, article, 'articles', 'provenance_title_only', candidate_specific=False)
        for name in entity['names'] + entity['aliases']:
            if not safe_name(name):
                continue
            for n in claim_items:
                if name_matches(n['surface_form'], name):
                    attach(entity['node_ids'][0], n['id'], 'explicit_mentions', 'safe_literal_mention')
        for row in sorted(links[eid].values(), key=lambda x: (FIELDS.index(x['field']), x['evidence_id'])):
            entity['evidence'][row['field']].append(row)
    owners = defaultdict(list)
    for eid, entity in entities.items():
        for field, rows in entity['evidence'].items():
            for row in rows:
                owners[row['evidence_id']].append(dict(entity_id=eid, field=field, support_kind=row['support_kind']))
    for key in arcs:
        arcs[key].sort(key=lambda a: (a['relation'], a['destination_node'], a['hyperedge_id']))
    return dict(snapshot=cutoff, entities=entities, node_to_entity=node_to_entity, items=items,
        evidence_owners=dict(owners), arcs=dict(arcs), rejected_edges=rejected,
        schema_policy='Unique typed subject/targets, never arbitrary co-membership; citation paths are structural only')


def expand_graph(index, seeds, eligible, cfg):
    """At most two directed edges, with article as the only intermediate node."""
    scores, reasons = {}, defaultdict(list)
    entities = index['node_to_entity']
    for source, score in seeds[:cfg['graph_seed_depth']]:
        first = index['arcs'].get(source, [])
        for step in first[:cfg['graph_fanout_cap']]:
            destination = step['destination_node']
            paths = [[step]]
            item = index['items'].get(destination)
            next_arcs = index['arcs'].get(destination, [])
            if cfg['graph_max_hops'] >= 2 and item and item['type'] == 'article' and len(next_arcs) <= cfg['graph_article_degree_cap']:
                paths += [[step, second] for second in next_arcs[:cfg['graph_fanout_cap']]
                          if second['relation'] in {'presents', 'cites'} and second['destination_node'] != source]
            for path in paths:
                dest = path[-1]['destination_node']
                eid = entities.get(dest)
                if eid not in eligible:
                    continue
                value = float(score) * cfg['graph_path_decay'] ** (len(path)-1)
                scores[eid] = max(scores.get(eid, -1.), value)
                trace = dict(source_evidence_id=source, destination_entity=eid, destination_node=dest,
                    path_length=len(path), path=path, score=value, interpretation='structural_path_not_entailment')
                reasons[eid].append(trace)
    for eid in reasons:
        reasons[eid] = sorted(reasons[eid], key=lambda p: (-p['score'], p['source_evidence_id'], str(p['path'])))[:8]
    return scores, dict(reasons)


def select_evidence(index, eid, relevance, cfg, graph_paths=()):
    entity = index['entities'][eid]
    selected, omitted, seen = [], [], set()
    for field in FIELDS:
        rows = list(entity['evidence'][field])
        if field == 'graph_evidence':
            rows += [dict(evidence_id=p['source_evidence_id'], field=field, support_kind='structural_path_not_entailment',
                candidate_specific=False, paths=[p['path']]) for p in graph_paths]
        rows.sort(key=lambda r: (-float(relevance.get(r['evidence_id'], -1.)), not r['candidate_specific'], r['evidence_id']))
        count = 0
        for row in rows:
            item = index['items'][row['evidence_id']]
            key = normalize_name(item['text'])
            if key in seen:
                continue
            seen.add(key)
            value = dict(row, text=item['text'], node_id=item['node_id'], first_seen_year=item['first_seen_year'],
                provenance=item['provenance'], relevance=float(relevance.get(row['evidence_id'], -1.)))
            if count >= cfg['query_evidence_caps'][field]:
                omitted.append(dict(evidence_id=row['evidence_id'], field=field, reason='family_cap'))
                continue
            selected.append(value)
            count += 1
    identity = dict(field='identity', node_id=eid, text='Candidate: ' + entity['name'] + '\nType: ' + ', '.join(entity['types']))
    blocks = [identity] + [dict(field=r['field'], node_id=r['node_id'], evidence_id=r['evidence_id'],
        text=r['field'].replace('_', ' ') + ' [' + r['support_kind'] + ']: ' + r['text']) for r in selected]
    return dict(node_id=eid, entity_id=eid, name=entity['name'], selected=selected, omitted=omitted, blocks=blocks)
