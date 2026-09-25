"""Requirement support on existing candidate bundles; no evaluator or answer input."""
from pathlib import Path
import math,time
import numpy as np
from .io import digest,read_json
from .entailment import NLI
from .reranking import CandidateReranker
from .question_decomposition import candidate_hypothesis
from .evidence_attribution import classify_attribution


class ControlledLocalNLI(NLI):
    """Reuse the existing pinned offline NLI implementation and exact pair cache."""
    def __init__(self,cfg):
        super().__init__(cfg['nli_model'],cfg['nli_revision'])
        self.cache_path=Path(cfg['constraint_nli_cache'])/(digest(self.metadata)+'.json')
        self.cache=read_json(self.cache_path) if self.cache_path.exists() else {}

    def predict_controlled(self,pairs,batch_size=12):
        results=[]; start=time.perf_counter(); new=0; hits=0
        for first in range(0,len(pairs),256):
            batch=pairs[first:first+256]; missing={digest(list(p)) for p in batch if digest(list(p)) not in self.cache}
            try:
                values=self.predict(batch,batch_size)
                new+=len(missing); hits+=len(batch)-len(missing)
                results.extend(dict(status='computed',**v) for v in values)
            except (RuntimeError,ValueError,OSError) as error:
                results.extend(dict(status='failed',reason=type(error).__name__+': '+str(error)[:200],entailment=None,neutral=None,contradiction=None) for _ in batch)
        return results,dict(nli_pair_requests=len(pairs),nli_forward_pairs=new,nli_cache_hits=hits,nli_seconds=time.perf_counter()-start)


def aggregate_support(components,supports,violations,types,answer_type,cfg):
    """Core gate times geometric coverage: one strong secondary field cannot rescue a missing task."""
    positives=[s for c,s in zip(components,supports) if c['kind']!='exclusion']
    core=next((s for c,s in zip(components,supports) if c['kind']=='core_task'),0.)
    strength=float(np.mean(positives)) if positives else 0.
    geometric=math.prod(max(0.,min(1.,s)) for s in positives)**(1/len(positives)) if positives else 0.
    score=core*geometric
    exclusion_factor=math.prod(1-max(0.,min(1.,v)) for v in violations)
    # Unknown/free-form answer nouns receive no prior; no candidate is filtered.
    families={'method':{'method','technique','component','cited_work'},'model':{'method','component','cited_work'},
              'architecture':{'method','component','cited_work'},'dataset':{'dataset','cited_work'},'benchmark':{'dataset','metric','article'},
              'article':{'article','cited_work'},'paper':{'article','cited_work'},'metric':{'metric'},'technique':{'technique','method','cited_work'}}
    prior=cfg['constraint_type_mismatch_factor'] if answer_type in families and not set(types)&families[answer_type] else 1.
    threshold=cfg['constraint_diagnostic_support_threshold']
    return dict(score=score*exclusion_factor*prior,core_support=core,mean_support=strength,max_support=max(positives,default=0.),
        soft_coverage=geometric,requirement_coverage=sum(s>=threshold for s in positives)/len(positives) if positives else 0.,
        supported_components=sum(s>=threshold for s in positives),positive_components=len(positives),
        exclusion_factor=exclusion_factor,answer_type_prior=prior,score_components=dict(requirement_score=score*exclusion_factor*prior),
        coverage_definition='Diagnostic >=0.5 only; ranking uses threshold-free core times geometric mean, times exclusion/type factors')


class RequirementCosineReranker(CandidateReranker):
    def __init__(self,cfg,decomposition,similarities,names):
        self.cfg=cfg; self.decomposition=decomposition; self.similarities=similarities; self.names=names

    def score(self,question,candidate_name,evidence):
        d=self.decomposition; cfg=self.cfg; items=[]
        for field,values in evidence['fields'].items():
            for item in values:
                attribution=classify_attribution(item,self.names[evidence['entity_id']])
                items.append(dict(item,field=field,attribution=attribution,attribution_weight=cfg['constraint_attribution_weights'][attribution['category']]))
        requirements=[]; supports=[]; violations=[]; comparisons=0
        for component in d.components:
            compared=[]
            for item in items:
                sim=float(self.similarities[component['id']][item['node_id']]); comparisons+=1
                compared.append(dict(node_id=item['node_id'],text=item['text'],field=item['field'],attribution=item['attribution']['category'],
                    weight=item['attribution_weight'],cosine=sim,support=max(0.,sim)*item['attribution_weight']))
            compared.sort(key=lambda r:(-r['support'],-r['cosine'],r['node_id']))
            eligible=[r for r in compared if r['weight']>0]
            selected=eligible[:cfg['constraint_nli_top_evidence']]
            best=selected[0] if selected else None; value=best['support'] if best else 0.
            if component['kind']=='exclusion': violations.append(value)
            supports.append(value)
            requirements.append(dict(component=component,hypothesis=candidate_hypothesis(candidate_name,d,component),
                hypothesis_role='exclusion_violation' if component['kind']=='exclusion' else 'desired_support',
                evidence_comparisons=compared,nli_selected=selected,best_evidence=best,cosine_support=value,nli_support=None))
        score=aggregate_support(d.components,supports,violations,evidence['types'],d.answer_type,cfg)
        threshold=cfg['constraint_diagnostic_support_threshold']
        return dict(score,requirements=requirements,attributions=items,semantic_support_comparisons=comparisons,
            unsupported_requirements=[c['text'] for c,s in zip(d.components,supports) if c['kind']!='exclusion' and s<threshold],
            contradicted_requirements=[],no_direct_attributable_evidence=not any(i['attribution']['category'] in {'DIRECT_SUBJECT_SUPPORT','EXPLICIT_NAME_SUPPORT','COMPARISON_WINNER'} for i in items),
            status='computed',scorer='requirement_cosine')


class RequirementNLIReranker(CandidateReranker):
    """Prepared batch outputs via the same score interface; deterministic cosine fallback on failure."""
    def __init__(self,cfg,decomposition,cosine_details,nli_predictions):
        self.cfg=cfg; self.decomposition=decomposition; self.cosine=cosine_details; self.predictions=nli_predictions

    def score(self,question,candidate_name,evidence):
        source=self.cosine[evidence['entity_id']]; requirements=[]; supports=[]; violations=[]; failed=[]; contradicted=[]
        for request in source['requirements']:
            assessed=[]
            for item in request['nli_selected']:
                key=digest([item['text'],request['hypothesis']]); p=self.predictions.get(key)
                if p is None or p['status']!='computed': failed.append(key); assessed.append(dict(item,nli=p,support=None)); continue
                # Neutral is not entailment. Contradiction can reduce positive support.
                value=item['weight']*max(0.,p['entailment']-p['contradiction'])
                if request['hypothesis_role']=='exclusion_violation': value=item['weight']*p['entailment']
                assessed.append(dict(item,nli=p,support=value))
                if request['hypothesis_role']!='exclusion_violation' and p['prediction']=='contradiction': contradicted.append(request['component']['text'])
            valid=[i for i in assessed if i['support'] is not None]
            best=max(valid,key=lambda i:(i['support'],i['node_id']),default=None); value=best['support'] if best else 0.
            supports.append(value)
            if request['hypothesis_role']=='exclusion_violation': violations.append(value)
            requirements.append(dict(component=request['component'],hypothesis=request['hypothesis'],hypothesis_role=request['hypothesis_role'],
                nli_evidence=assessed,best_evidence=best,cosine_support=request['cosine_support'],nli_support=value))
        if failed:
            return dict(source,status='cosine_fallback_nli_failed',scorer='requirement_nli',nli_requirements=requirements,nli_failed_pair_keys=failed)
        score=aggregate_support(self.decomposition.components,supports,violations,evidence['types'],self.decomposition.answer_type,self.cfg)
        threshold=self.cfg['constraint_diagnostic_support_threshold']
        return dict(score,requirements=requirements,status='computed',scorer='requirement_nli',nli_failed_pair_keys=[],
            unsupported_requirements=[c['text'] for c,s in zip(self.decomposition.components,supports) if c['kind']!='exclusion' and s<threshold],
            contradicted_requirements=sorted(set(contradicted)),no_direct_attributable_evidence=source['no_direct_attributable_evidence'])


class PreparedReranker(CandidateReranker):
    def __init__(self,details): self.details=details
    def score(self,question,candidate_name,evidence): return self.details[evidence['entity_id']]


def uniform_nli_fallback(cosine,nli_details):
    """Never mix incomparable cosine and NLI scales after a partial execution failure."""
    failed=[ident for ident,d in nli_details.items() if d['status']!='computed']
    if not failed: return nli_details
    return {ident:dict(cosine[ident],status='cosine_fallback_nli_failed',scorer='requirement_nli',
        nli_requirements=d.get('nli_requirements',d['requirements']),nli_failed_candidates=failed,
        fallback_scope='whole_question_pool',nli_failed_pair_keys=d.get('nli_failed_pair_keys',[])) for ident,d in nli_details.items()}
