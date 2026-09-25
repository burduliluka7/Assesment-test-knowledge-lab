import csv
from pathlib import Path
import numpy as np

def human_ratings(path,sample):
    path=Path(path)
    if not path.exists(): return dict(status='missing_data',reason='Blind audit generated; no human ratings supplied')
    expected={f'A{i:04}':s for i,s in enumerate(sample)}; judged=[]; seen=set()
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            rating=row.get('rating','').strip().lower()
            if not rating: continue
            ident=row['audit_id']
            if ident not in expected or ident in seen: raise ValueError('Unknown or duplicate human audit id')
            if row.get('gloss')!=expected[ident]['gloss'] or row.get('label')!=expected[ident]['label']:
                raise ValueError('Human audit does not match this generated sample')
            if rating not in ['accurate','vague','unsupported','wrong']: raise ValueError('Invalid human audit category')
            seen.add(ident); judged.append((expected[ident]['snapshot'],expected[ident]['level'],rating))
    grouped={}
    for year,level,_ in judged:
        key=f'{year}:L{level}'
        if key not in grouped:
            rr=[r for y,k,r in judged if y==year and k==level]
            grouped[key]=dict(n=len(rr),overclaim_rate=sum(r in ['unsupported','wrong'] for r in rr)/len(rr))
    return dict(status='computed' if judged else 'missing_data',n=len(judged),by_snapshot_level=grouped,
        rule='(unsupported + wrong) / evaluated; vague retained in denominator')

def evaluate(logs,cfg,outdir):
    rng=np.random.default_rng(cfg['seed']); sample=[]
    for snapshot in sorted({x['snapshot'] for x in logs}):
        for level in [0,1]:
            group=[x for x in logs if x['snapshot']==snapshot and x['level']==level]
            if group:
                sample.extend(group[i] for i in rng.permutation(len(group))[:cfg['faithfulness_sample_per_level']])
    rng.shuffle(sample); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/'audit_blind.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['audit_id','label','gloss','evidence','rating','notes']); writer.writeheader()
        for i,s in enumerate(sample): writer.writerow(dict(audit_id=f'A{i:04}',label=s['label'],gloss=s['gloss'],evidence=' | '.join(n['surface_form'] for n in s['input'])))
    key=[dict(audit_id=f'A{i:04}',cluster_id=s['id'],snapshot=s['snapshot'],level=s['level']) for i,s in enumerate(sample)]
    from ..io import write_json
    write_json(outdir/'audit_key.json',key)
    (outdir/'audit_instructions.md').write_text('Rate each gloss against its evidence: accurate, vague, unsupported, or wrong. Keep variant/cluster identities hidden while rating. Copy audit_blind.csv to audit_ratings.csv and fill rating cells; do not alter labels or glosses. Re-evaluation imports matching ratings and computes (unsupported + wrong) / evaluated. No ratings are generated automatically.\n',encoding='utf-8')
    result=dict(human_audit=human_ratings(outdir/'audit_ratings.csv',sample),
        independent_evidence=dict(status='missing_data',reason='No independent source-paper full text supplied; benchmark quotations are held out for retrieval and never used for labels'),
        nli_proxy=dict(status='missing_model'),sample_size=len(sample))
    if not sample:
        result['nli_proxy']=dict(status='not_applicable',reason='no non-singleton labels'); return result
    try:
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        torch.set_num_threads(4)
        tok=AutoTokenizer.from_pretrained(cfg['nli_model'],revision=cfg['nli_revision'])
        model=AutoModelForSequenceClassification.from_pretrained(cfg['nli_model'],revision=cfg['nli_revision'],use_safetensors=True).eval()
        labelmap={int(i):str(x).lower() for i,x in model.config.id2label.items()}
        if not any('entail' in x for x in labelmap.values()):
            raise ValueError(f'NLI id2label is not explicit: {labelmap}')
        ei=next(i for i,x in labelmap.items() if 'entail' in x)
        rows=[]
        for s in sample:
            # Independent model, shared evidence: a support proxy, never an independent truth audit.
            relevant=sorted(s['input'],key=lambda n:(-sum(k in n['surface_form'].lower() for k in s['keywords']),n['id']))[:8]
            premise=' '.join(n['surface_form'] for n in relevant)
            inp=tok(premise,s['gloss'],return_tensors='pt',truncation='only_first',max_length=512)
            with torch.inference_mode(): probs=model(**inp).logits.softmax(-1)[0].numpy()
            rows.append(dict(cluster_id=s['id'],snapshot=s['snapshot'],level=s['level'],
                probabilities={labelmap[i]:float(p) for i,p in enumerate(probs)},
                overclaim_proxy=bool(probs[ei]<cfg['nli_threshold']),premise_member_ids=[n['id'] for n in relevant],premise=premise))
        grouped={}
        for year in sorted({r['snapshot'] for r in rows}):
            for level in [0,1]:
                group=[r for r in rows if r['snapshot']==year and r['level']==level]
                if group: grouped[f'{year}:L{level}']=dict(n=len(group),overclaim_rate=sum(r['overclaim_proxy'] for r in group)/len(group))
        result['nli_proxy']=dict(status='computed',model=cfg['nli_model'],revision=cfg['nli_revision'],
            rule=f"non-entailment: entailment probability < {cfg['nli_threshold']}; includes neutral and contradiction",
            limitation='Shared generator evidence, truncated premises, no scientific truth validation; may penalize vague labels. Human overclaim rate remains unmeasured.',
            by_snapshot_level=grouped,rows=rows)
    except (OSError,ValueError,ImportError,RuntimeError) as e:
        result['nli_proxy']=dict(status='missing_model',reason=str(e),model=cfg['nli_model'])
    return result
