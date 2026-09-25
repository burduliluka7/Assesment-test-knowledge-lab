"""Frozen conditioned scoring and budgeted search. No evaluation inputs here."""
from copy import deepcopy
import heapq
import numpy as np
from .conditioned_decomposition import decompose_conditioned_question
from .conditioned_evidence import ConditionedEvidenceReranker
from .search_v2 import SearchIndex
from .retrieval import normalize


def final_decomposition(question):
    d = decompose_conditioned_question(question)
    # Preserve extracted qualifiers in source_spans for diagnostics only.
    d.components = [c for c in d.components if c['kind'] != 'answer_qualifier']
    d.answer_qualifiers = []
    return d


class FineScorer:
    """Lazy per-candidate scoring; each system pays its standalone logical work.

    Similarity caches can reproduce archived float32 arithmetic. Cached execution
    across systems is recorded separately and is not claimed as a speedup.
    """
    def __init__(self, question, cfg, pools, names, vectors, node_vectors, similarities=None):
        self.question = question
        self.decomposition = final_decomposition(question)
        self.cfg, self.pools, self.names = cfg, pools, names
        self.vectors, self.node_vectors = vectors, node_vectors
        self.similarities = similarities or {}
        self.details = {}
        self.actual_new_dots = 0
        self.calls = []

    def work(self, entity_id):
        return len(self.decomposition.components) * sum(map(len, self.pools[entity_id]['fields'].values()))

    def score(self, entity_id):
        self.calls.append(entity_id)
        if entity_id not in self.details:
            ids = sorted({i['node_id'] for xs in self.pools[entity_id]['fields'].values() for i in xs})
            sims = {}
            for c in self.decomposition.components:
                cache = self.similarities.setdefault(c['text'], {})
                missing = [n for n in ids if n not in cache]
                if missing:
                    dots = self.vectors[c['text']] @ np.asarray([self.node_vectors[n] for n in missing]).T
                    cache.update(zip(missing, map(float, dots)))
                    self.actual_new_dots += len(missing)
                sims[c['id']] = cache
            scorer = ConditionedEvidenceReranker(self.cfg, self.decomposition, sims, self.names)
            pool = self.pools[entity_id]
            self.details[entity_id] = scorer.score(self.question, pool['name'], pool)
        return self.details[entity_id]


def rank_scored(ids, scorer):
    rows = []
    for ident in ids:
        p = scorer.pools[ident]; d = scorer.score(ident)
        rows.append(dict(entity_id=ident, name=p['name'], node_ids=p['node_ids'], types=p['types'],
                         score=d['score'], score_components=d['score_components']))
    return sorted(rows, key=lambda r: (-r['score'], r['entity_id']))


class HierarchySearch:
    def __init__(self, hierarchy, nodes, X, groups, prototypes=4):
        self.group_by_node = {n:g['id'] for g in groups for n in g['node_ids']}
        self.group_members = {g['id']:g['node_ids'] for g in groups}
        # Groups may contain mixed-type exact-name duplicates. A union of their
        # types would incorrectly admit every unrelated node of those extra types.
        # Mask only eligibility; cached original text vectors and hierarchy stay fixed.
        masked = [dict(n, type='candidate' if n['id'] in self.group_by_node else 'excluded') for n in nodes]
        self.index = SearchIndex(hierarchy, masked, X, {'candidate'}, prototypes)
        self.level = {c['id']:l for l, cs in enumerate(hierarchy['levels']) for c in cs}
        self.leaf_level = len(hierarchy['levels']) - 1
        # Singleton intermediate clusters still have descendants; do not terminate
        # traversal early merely because an intermediate cluster has one member.
        for uid, u in self.index.units.items():
            u['leaf'] = self.level[uid] == self.leaf_level
        self.roots = sorted(self.index.children[None])
        self.root_cost = sum(self.charge(u) for u in self.roots)

    def charge(self, uid):
        return 1 + len(self.index.units[uid]['representatives'])

    def search(self, query, budget, scorer, allowed_groups=None, alpha=.5, top_r=2):
        q = normalize(query); ix = self.index; trace = []; expanded = []; reached = []
        seen_groups = set(); selected = []; coarse = leaf = evidence = 0
        limit = float('inf') if budget is None else budget
        if limit < self.root_cost:
            return self.result([], budget, 'INSUFFICIENT_ROOT_BUDGET', [], [], [], 0, 0, 0)

        def coarse_score(uid):
            nonlocal coarse
            u = ix.units[uid]; pp = sorted((float(ix.X[i] @ q) for i in u['representatives']), reverse=True)
            v = float(u['centroid'] @ q)
            if pp: v = alpha*v + (1-alpha)*float(np.mean(pp[:top_r]))
            coarse += self.charge(uid)
            trace.append(dict(object_id=uid, level=self.level[uid], action='score_cluster', comparisons=self.charge(uid), score=v))
            return v

        heap = [(-coarse_score(uid), uid) for uid in self.roots]; heapq.heapify(heap)
        while heap:
            _, uid = heapq.heappop(heap); expanded.append(uid)
            trace.append(dict(object_id=uid, level=self.level[uid], action='expand', comparisons=0))
            for child in ix.children[uid]:
                u = ix.units[child]
                if not u['leaf']:
                    if coarse + leaf + evidence + self.charge(child) <= limit:
                        heapq.heappush(heap, (-coarse_score(child), child))
                    continue
                if coarse + leaf + evidence + 1 > limit: continue
                node = u['member_id']; leaf += 1; reached.append(node)
                trace.append(dict(object_id=child, node_id=node, parent_id=uid, level=self.level[child], action='score_leaf', comparisons=1,
                                  score=float(u['centroid'] @ q)))
                ident = self.group_by_node.get(node)
                if ident is None or ident in seen_groups or allowed_groups is not None and ident not in allowed_groups: continue
                charge = scorer.work(ident)
                if coarse + leaf + evidence + charge > limit: continue
                # One reached atomic member activates its pre-existing exact-name
                # group. The evidence pool is unchanged; this is explicitly logged.
                seen_groups.add(ident); selected.append(ident); evidence += charge
                scorer.score(ident)
                trace.append(dict(entity_id=ident, reached_via=node, action='score_candidate', comparisons=charge))
        return self.result(rank_scored(selected, scorer), budget, 'COMPUTED', trace, expanded, reached, coarse, leaf, evidence)

    def result(self, ranked, budget, status, trace, expanded, reached, coarse, leaf, evidence):
        reached_set = set(reached)
        lifted = [dict(entity_id=r['entity_id'], member_count=len(r['node_ids']),
                       reached_member_ids=sorted(set(r['node_ids'])&reached_set),
                       reached_fraction=len(set(r['node_ids'])&reached_set)/len(r['node_ids'])) for r in ranked]
        return dict(ranked=ranked, budget=budget, status=status, root_layer_cost=self.root_cost, root_layer_complete=status=='COMPUTED',
                    trace=trace, expanded_clusters=expanded, reached_node_ids=reached, group_lift=lifted,
                    cost=dict(coarse_comparisons=coarse, leaf_comparisons=leaf, evidence_comparisons=evidence,
                              fine_candidates_scored=len(ranked), total_comparisons=coarse+leaf+evidence,
                              cluster_expansions=len(expanded), reached_atomic_candidates=len(set(reached))))


def flat_scan(scorer, query, node_vectors, budget=None, seed=42):
    ids = sorted(scorer.pools)
    if budget is not None: ids = np.random.default_rng(seed).permutation(ids).tolist()
    q = normalize(query); selected = []; cost = evidence = 0; trace = []
    for ident in ids:
        work = 1 + scorer.work(ident)
        # Prefix scan: do not choose cheaper candidates after seeing their cost.
        if budget is not None and cost+work > budget: break
        pool = scorer.pools[ident]
        similarity = float(node_vectors[pool['node_ids'][0]] @ q)
        selected.append(ident); cost += work; evidence += work-1
        trace.append(dict(entity_id=ident, comparisons=work, leaf_cosine=similarity))
    return dict(ranked=rank_scored(selected, scorer), budget=budget, status='COMPUTED', trace=trace,
                cost=dict(coarse_comparisons=0, leaf_comparisons=len(selected), evidence_comparisons=evidence,
                          fine_candidates_scored=len(selected), total_comparisons=cost, cluster_expansions=0,
                          reached_atomic_candidates=len(selected)))
