"""Independent channels, deterministic set union, rank-only fusion, and provenance."""
from collections import defaultdict
import numpy as np
from .entity_index import allowed_entity, name_matches, expand_graph
from .ranking import BM25


def ranked(scores):
    return [dict(entity_id=i, score=float(s)) for i, s in sorted(scores.items(), key=lambda p: (-float(p[1]), p[0]))]


def round_robin(channels):
    result, seen = [], set()
    for pos in range(max((len(rows) for rows in channels.values()), default=0)):
        for rows in channels.values():
            if pos < len(rows) and rows[pos]['entity_id'] not in seen:
                ident = rows[pos]['entity_id']
                seen.add(ident)
                result.append(dict(entity_id=ident, score=1. / (1 + len(result)), stage='round_robin_union'))
    return result


def fuse(channels, k=60):
    values, contributions = defaultdict(float), defaultdict(dict)
    for name in sorted(channels):
        rows = channels[name]
        for rank_number, row in enumerate(rows, 1):
            ident = row['entity_id']
            contribution = 1. / (k + rank_number)
            values[ident] += contribution
            contributions[ident][name] = dict(rank=rank_number, score=row['score'], contribution=contribution)
    return ranked(values), dict(contributions)


class EntityRetriever:
    def __init__(self, index, documents, encoder, cfg):
        self.index, self.cfg, self.encoder = index, cfg, encoder
        self.ids = sorted(index['entities'])
        self.docs = documents
        self.doc_ids = [d['node_id'] for d in documents]
        self.doc_index = {i: j for j, i in enumerate(self.doc_ids)}
        self.doc_vectors = encoder.encode([d['text'] for d in documents])
        self.evidence_ids = sorted(index['items'])
        self.evidence_vectors = encoder.encode([index['items'][i]['text'] for i in self.evidence_ids])
        self.lexical_texts = []
        for eid in self.ids:
            entity = index['entities'][eid]
            evidence = list(dict.fromkeys(r['evidence_id'] for rows in entity['evidence'].values() for r in rows))
            # All indexed evidence is retained for lexical recall; each node once.
            self.lexical_texts.append('\n'.join([*entity['names'], *entity['aliases'], *[index['items'][i]['text'] for i in evidence]]))
        self.bm25 = BM25(self.lexical_texts, cfg['bm25_k1'], cfg['bm25_b'])

    def generate(self, intent):
        cfg, index = self.cfg, self.index
        allowed = {i for i in self.ids if allowed_entity(index['entities'][i], intent, cfg)}
        texts = [intent['original_question'], intent['structured_query'], intent['parsed_only_query']]
        queries = self.encoder.encode(texts)
        dense = self.doc_vectors @ queries.T
        evidence = self.evidence_vectors @ queries[:2].T
        evidence_max = evidence.max(axis=1)
        evidence_similarity = dict(zip(self.evidence_ids, map(float, evidence_max)))
        full_channels = {}
        for view, name in enumerate(('dense_original', 'dense_structured', 'dense_parsed_only')):
            full_channels[name] = ranked({eid: max(float(dense[self.doc_index[n], view]) for n in index['entities'][eid]['node_ids']) for eid in allowed})
        dense_maps = {name: {r['entity_id']: r['score'] for r in full_channels[name]} for name in ('dense_original', 'dense_structured')}
        combined = ranked({i: max(dense_maps[name][i] for name in dense_maps) for i in allowed})
        lexical_scores = self.bm25.score(intent['original_question'] + '\n' + intent['structured_query'])
        # Zero-overlap lexical rows never receive arbitrary RRF votes.
        full_channels['bm25'] = ranked({i: float(s) for i, s in zip(self.ids, lexical_scores) if i in allowed and s > 0})
        matches = {eid: [n for n in index['entities'][eid]['names'] + index['entities'][eid]['aliases'] if name_matches(intent['original_question'], n)] for eid in allowed}
        full_channels['name'] = ranked({i: max(len(n) for n in names) for i, names in matches.items() if names})
        evidence_scores, evidence_reasons = {}, defaultdict(list)
        seeds = sorted(evidence_similarity.items(), key=lambda p: (-p[1], p[0]))
        for ev, value in seeds[:cfg['evidence_seed_depth']]:
            for owner in index['evidence_owners'].get(ev, []):
                eid = owner['entity_id']
                if eid not in allowed:
                    continue
                evidence_scores[eid] = max(evidence_scores.get(eid, -1.), value)
                evidence_reasons[eid].append(dict(evidence_id=ev, similarity=value, **owner))
        full_channels['evidence'] = ranked(evidence_scores)
        graph_scores, graph_reasons = expand_graph(index, seeds, allowed, cfg)
        full_channels['graph'] = ranked(graph_scores)
        channel_order = ('dense_original', 'dense_structured', 'bm25', 'name', 'evidence', 'graph')
        channels = {name: full_channels[name][:cfg['channel_depth']] for name in channel_order}
        stage_names = {'A2': channel_order[:4], 'A3': channel_order[:5], 'A4': channel_order}
        systems = {'A1': combined[:cfg['channel_depth']],
            'TypedRaw': full_channels['dense_original'][:cfg['channel_depth']],
            'ParsedOnly': full_channels['dense_parsed_only'][:cfg['channel_depth']]}
        for name, included in stage_names.items():
            systems[name] = round_robin({c: channels[c] for c in included})
        systems['A5'], contributions = fuse(channels, cfg['rrf_k'])
        systems['WithoutEvidenceRRF'], _ = fuse({c: r for c, r in channels.items() if c != 'evidence'}, cfg['rrf_k'])
        systems['WithoutGraphRRF'], _ = fuse({c: r for c, r in channels.items() if c != 'graph'}, cfg['rrf_k'])
        baseline = {}
        for nid, value in zip(self.doc_ids, dense[:, 0]):
            ident = index['node_to_entity'].get(nid, 'node:' + nid)
            baseline[ident] = max(baseline.get(ident, -1.), float(value))
        systems['A0'] = ranked(baseline)[:cfg['channel_depth']]
        sources = {i: dict(channels=contributions.get(i, {}), name_matches=matches.get(i, []),
            evidence=evidence_reasons.get(i, [])[:8], graph=graph_reasons.get(i, [])) for i in sorted(contributions)}
        costs = dict(dense_comparisons=len(self.doc_ids) * len(queries), evidence_comparisons=len(self.evidence_ids) * 2,
            bm25_document_term_lookups=len(self.ids) * len(set(__import__('re').findall(r'\w+', texts[0] + texts[1]))),
            name_entity_checks=len(allowed), graph_seed_nodes=min(len(seeds), cfg['graph_seed_depth']),
            candidate_union_size=len(systems['A5']), eligible_entities=len(allowed))
        return dict(systems=systems, channels=channels, full_channels=full_channels, candidate_sources=sources,
            evidence_similarity=evidence_similarity, graph_paths=graph_reasons, costs=costs)


def rerank_prefix(rows, score_details, k):
    prefix = rows[:k]
    scores = {r['entity_id']: score_details[r['entity_id']]['relevance_score'] for r in prefix}
    assert len(scores) == len(prefix)
    head = [dict(r, stage='relevance_cross_encoder') for r in ranked(scores)]
    return head + [dict(r, stage='unreranked_fusion_tail') for r in rows[k:]]
