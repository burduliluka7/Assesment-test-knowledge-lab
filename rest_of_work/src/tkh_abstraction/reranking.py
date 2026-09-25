"""Local candidate scoring only; oracle and expected targets are absent here."""
from abc import ABC,abstractmethod
from pathlib import Path
import math,json,time
import numpy as np
from .diffusion import rrf_fuse


class CandidateReranker(ABC):
    @abstractmethod
    def score(self,question,candidate_name,evidence): raise NotImplementedError


class EvidenceSemanticReranker(CandidateReranker):
    def __init__(self,cfg):
        self.cfg=cfg
        weights=cfg['reranker_field_weights']
        if any(v<0 for v in weights.values()) or weights.get('name',0)<=0: raise ValueError('Nonnegative weights and positive name weight required')
        if cfg['reranker_pooling'] not in {'max','top2_mean'}: raise ValueError('Unknown pooling')
        if not 0<=cfg['reranker_coverage_floor']<=1: raise ValueError('Coverage floor must be in [0,1]')

    def score(self,question,candidate_name,evidence):
        weights=self.cfg['reranker_field_weights']; total=sum(weights.values()); available={}; weighted={}
        available['name']=max(0.,evidence['name_similarity'])
        for field,items in evidence['fields'].items():
            if not items: continue
            values=sorted((max(0.,x['similarity']) for x in items),reverse=True)
            available[field]=values[0] if self.cfg['reranker_pooling']=='max' else float(np.mean(values[:2]))
        mass=sum(weights[f] for f in available)
        for field,sim in available.items(): weighted[field]=weights[field]*sim
        semantic=sum(weighted.values())/mass if mass else 0.; coverage=mass/total if total else 0.
        floor=self.cfg['reranker_coverage_floor']; adjustment=floor+(1-floor)*math.sqrt(coverage)
        score=semantic*adjustment
        return dict(score=score,available_semantic_score=semantic,coverage=coverage,coverage_adjustment=adjustment,
            available_weight=mass,total_weight=total,field_scores=available,weighted_field_scores=weighted,
            score_components={f:v/mass*adjustment for f,v in weighted.items()} if mass else {})


def inspect_local_reranker(explicit_path=None):
    """Inspect cached configurations only. Never resolve or download a Hub model."""
    from huggingface_hub.constants import HF_HUB_CACHE
    roots=[Path(explicit_path)] if explicit_path else sorted(Path(HF_HUB_CACHE).glob('models--*/snapshots/*'))
    entries=[]
    for root in roots:
        config=root/'config.json'
        if not config.is_file(): continue
        d=json.loads(config.read_text(encoding='utf-8')); arch=d.get('architectures',[]); labels=d.get('id2label',{})
        classification=any('ForSequenceClassification' in a for a in arch)
        nli={'entailment','neutral','contradiction'}<=set(str(x).lower() for x in labels.values())
        weights=bool(list(root.glob('*.safetensors')))
        relevance_hint=any(t in str(root).casefold() for t in ['rerank','ms-marco','ms_marco'])
        suitable=classification and not nli and weights and (bool(explicit_path) or relevance_hint)
        entries.append(dict(path=str(root),architectures=arch,labels=labels,weights_present=weights,suitable=suitable,
            reason='NLI classification is not relevance reranking' if nli else 'No local relevance-trained sequence-classifier identified' if not suitable else 'Local relevance model candidate'))
    suitable=[e for e in entries if e['suitable']]
    return dict(status='available' if suitable else 'unavailable',selected_path=suitable[0]['path'] if suitable else None,inspected=entries,
        reason=None if suitable else 'No suitable local relevance reranker present; cached NLI model is not substituted',network_access=False)


class LocalCrossEncoderReranker(CandidateReranker):
    def __init__(self,path,token_budget=480,relevance_label=None):
        path=Path(path)
        if not path.is_dir(): raise FileNotFoundError('Cross-encoder must be an existing local directory')
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        import torch
        torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(str(path),local_files_only=True,use_safetensors=True).eval()
        self.token_budget=min(token_budget,int(getattr(self.model.config,'max_position_embeddings',512)))
        self.labels={int(k):str(v) for k,v in self.model.config.id2label.items()}; self.relevance_label=relevance_label
        if self.model.config.num_labels!=1 and relevance_label not in self.labels.values():
            raise ValueError('Multiclass reranker requires an explicit relevance label; never guess NLI relevance')

    def score(self,question,candidate_name,evidence):
        import torch
        included=[]; omitted=[]; document=''
        for block in evidence['document_blocks']:
            trial=(document+'\n'+block['text']).strip()
            tokens=len(self.tokenizer(question,trial,add_special_tokens=True)['input_ids'])
            if tokens<=self.token_budget: included.append(block); document=trial
            else:
                omitted=[dict(b,reason='priority_prefix_token_budget') for b in evidence['document_blocks'][len(included):]]
                break
        if not included or included[0]['field']!='name': raise ValueError('Question and candidate identity exceed token budget')
        inputs=self.tokenizer(question,document,truncation=False,return_tensors='pt')
        with torch.inference_mode(): logits=self.model(**inputs).logits[0]
        index=0 if len(logits)==1 else next(i for i,label in self.labels.items() if label==self.relevance_label)
        value=float(logits[index]) if len(logits)==1 else float(logits.softmax(-1)[index])
        return dict(score=value,score_components=dict(cross_encoder=value),token_count=int(inputs['input_ids'].shape[1]),included=included,omitted=omitted)


def rerank_candidates(frozen_ranking,bundles,reranker,question,candidate_k,fuse=False,rrf_k=60):
    prefix=frozen_ranking[:candidate_k]; start=time.perf_counter(); scored=[]; details={}
    for row in prefix:
        detail=reranker.score(question,row['name'],bundles[row['entity_id']]); details[row['entity_id']]=detail
        scored.append(dict(row,score=detail['score'],score_components=detail['score_components']))
    scored.sort(key=lambda r:(-r['score'],r['entity_id']))
    rerank_seconds=time.perf_counter()-start
    if fuse:
        scored=rrf_fuse(prefix,scored,rrf_k)
        for row in scored:
            row['score_components']={'rrf_h2':row['score_components']['rrf_r3'],'rrf_reranker':row['score_components']['rrf_diffusion']}
    if {r['entity_id'] for r in scored}!={r['entity_id'] for r in prefix}: raise ValueError('Reranker changed candidate membership')
    return dict(ranked=scored+list(frozen_ranking[len(prefix):]),details=details,comparisons=len(prefix),
        runtime_seconds=time.perf_counter()-start,scoring_seconds=rerank_seconds,candidate_preserved=True)
