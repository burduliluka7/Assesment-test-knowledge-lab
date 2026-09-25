"""Query-independent graph pools and component selection using the existing CR1 aggregator."""
from dataclasses import dataclass, asdict
from copy import deepcopy
from .candidate_evidence import FIELD_ORDER
from .target_mapping import normalize
from .constraint_scoring import RequirementCosineReranker, aggregate_support
from .conditioned_decomposition import classify_conditioned_attribution
from .evidence_attribution import classify_attribution
from .question_decomposition import candidate_hypothesis


@dataclass
class CandidateEvidencePool:
    entity_id: str
    name: str
    types: list
    node_ids: list
    cutoff: int
    fields: dict
    available_counts: dict
    discarded: list
    weighted_degree: float

    def to_dict(self): return asdict(self)


def build_candidate_evidence_pool(index, entity_id, caps):
    """No question argument. Reuse eligible graph pools before whole-question selection.

    No extra path traversal: all safe named claims are already in MentionIndex.
    Historical query-seeded paths remain separate diagnostics, not pool inputs.
    """
    group = index.groups[entity_id]; fields = {}; discarded = []; available = {}
    used = {normalize(index.nodes[n]['surface_form']): None for n in group['node_ids']}
    for field in FIELD_ORDER:
        unique = {}
        for item in sorted(index.pools[entity_id][field], key=lambda x:(x['priority'], x.get('support_kind')=='comembership_only', x['node_id'], str(x['path']))):
            key = normalize(item['text'])
            provenance = {k:deepcopy(item.get(k)) for k in ('node_id','kind','path','provenance','support_kind','edge_arity')}
            if key in used:
                if used[key] is not None: used[key]['source_links'].append(provenance)
                continue
            if key not in unique:
                names=index.mentions.names[entity_id]
                unique[key] = dict(deepcopy(item), field=field, source_links=[],
                                   attribution=classify_conditioned_attribution(item,names),
                                   historical_attribution=classify_attribution(item,names))
            unique[key]['source_links'].append(provenance)
        available[field] = len(unique)
        chosen = list(unique.items())[:caps[field]]
        fields[field] = [v for _,v in chosen]
        used.update(chosen)
        discarded.extend(dict(node_id=v['node_id'], field=field, reason='structural_field_cap') for _,v in list(unique.items())[caps[field]:])
    return CandidateEvidencePool(group['id'], group['name'], group['types'], group['node_ids'], index.snap['snapshot'], fields,
        available, discarded, sum(float(index.op.degree[index.op.idx[n]]) for n in group['node_ids']))


def select_evidence_for_component(component, items, similarities, top_k=3):
    compared = []
    for item in items:
        cosine = float(similarities[component['id']][item['node_id']]); weight = item['attribution_weight']
        compared.append(dict(node_id=item['node_id'], text=item['text'], field=item['field'],
            attribution=item['attribution']['category'], weight=weight, cosine=cosine, support=max(0.,cosine)*weight))
    compared.sort(key=lambda r:(-r['support'], -r['cosine'], r['node_id']))
    return [r for r in compared if r['weight']>0][:top_k], compared


class ConditionedEvidenceReranker(RequirementCosineReranker):
    """Same interface and aggregation; new evidence access, with explicit grammar version."""
    def __init__(self, cfg, decomposition, similarities, names, corrected_attribution=True):
        super().__init__(cfg, decomposition, similarities, names)
        self.classifier = classify_conditioned_attribution if corrected_attribution else classify_attribution

    def score(self, question, candidate_name, evidence):
        items=[]
        for field, values in evidence['fields'].items():
            for item in values:
                attr=item.get('attribution' if self.classifier is classify_conditioned_attribution else 'historical_attribution')
                if attr is None: attr=self.classifier(item,self.names[evidence['entity_id']])
                items.append(dict(item, field=field, attribution=attr,
                                  attribution_weight=self.cfg['constraint_attribution_weights'][attr['category']]))
        requirements=[]; supports=[]; violations=[]; comparisons=0
        for c in self.decomposition.components:
            selected, compared=select_evidence_for_component(c,items,self.similarities,self.cfg['conditioned_evidence_top_k'])
            comparisons+=len(compared); best=selected[0] if selected else None; value=best['support'] if best else 0.
            supports.append(value)
            if c['kind']=='exclusion': violations.append(value)
            requirements.append(dict(component=c, best_evidence=best, selected_evidence=selected, cosine_support=value,
                maximum_raw_cosine=max((r['cosine'] for r in compared),default=None),
                hypothesis=candidate_hypothesis(candidate_name,self.decomposition,c), hypothesis_used_for_scoring=False))
        score=aggregate_support(self.decomposition.components,supports,violations,evidence['types'],self.decomposition.answer_type,self.cfg)
        return dict(score, requirements=requirements, semantic_support_comparisons=comparisons,
                    no_direct_attributable_evidence=not any(i['attribution']['category'] in {'DIRECT_SUBJECT_SUPPORT','EXPLICIT_NAME_SUPPORT','COMPARISON_WINNER'} for i in items),
                    pool_size=len(items), attributable_items=sum(i['attribution_weight']>0 for i in items),
                    attribution_changes=[dict(node_id=i['node_id'], old=old['category'], new=i['attribution']['category']) for i in items
                        for old in [i.get('historical_attribution') or classify_attribution(i,self.names[evidence['entity_id']])]
                        if old['category']!=i['attribution']['category']],
                    status='computed', scorer='requirement_conditioned_cosine')
