"""Post-hoc NLI controls, evaluated separately and never used to score candidates."""
from pathlib import Path
from tkh_abstraction.config import read_config
from tkh_abstraction.io import read_json,write_json,digest
from tkh_abstraction.constraint_scoring import ControlledLocalNLI

root=Path('artifacts/evaluation'); support=read_json(root/'requirement_support.json'); attrs=read_json(root/'evidence_attribution.json')
records=[]; seen=set()
for a in attrs:
    if a['cutoff']!=2025 or a['candidate'].casefold() not in {'m3gnet','schnet'}: continue
    for item in a['items']:
        if item['attribution']['category']!='COMPARISON_BASELINE': continue
        key=(a['candidate'],item['text'])
        if key in seen: continue
        seen.add(key)
        records.append(dict(kind='actual_comparison_baseline',premise=item['text'],hypothesis=f'{a["candidate"]} outperforms the other model.',node_id=item['node_id']))
        if len(records)>=4: break
    if len(records)>=4: break
# Names are diagnostic fixtures, not production rules. The first is an actual graph claim.
nodes={n['id']:n for n in read_json('data/data/tkh_collection10.json')['nodes']}
premise=nodes['clai_00343']['surface_form']
for name in ['SevenNet','M3GNet']:
    records.append(dict(kind='paired_actual_comparison_subject',premise=premise,hypothesis=f'{name} outperforms the other model.',node_id='clai_00343'))
for text in [
    'D4FT converges on carbon fragments containing 120 and 140 atoms.',
    'D4FT converges on carbon fragments containing more than one million atoms.']:
    records.append(dict(kind='technical_scale_control',premise=nodes['clai_00430']['surface_form'],hypothesis=text,node_id='clai_00430'))
# Identity/negation controls from actual scientific assertions. Expected entailment of
# an identical premise is a logical control, not a scientific endorsement of the claim.
for ident in ['clai_00343','clai_00430','clai_00646']:
    text=nodes[ident]['surface_form']
    records.append(dict(kind='technical_identity_control',premise=text,hypothesis=text,node_id=ident))
    records.append(dict(kind='technical_negation_control',premise=text,hypothesis='It is not true that '+text[0].lower()+text[1:],node_id=ident))
# Actual high-confidence entailments and negative controls with a mismatched graph premise.
for q in support:
    if q['cutoff']!=2025: continue
    options=[]
    for c in q['candidates'][:100]:
        for req in c['CR2']['requirements']:
            for e in req.get('nli_evidence',[]):
                if e.get('nli') and e['nli']['status']=='computed': options.append((e['nli']['entailment'],e,req))
    if not options: continue
    _,e,req=max(options,key=lambda x:x[0]); records.append(dict(kind='actual_high_entailment',question_id=q['question_id'],premise=e['text'],hypothesis=req['hypothesis']))
    alternative=next((a for a in attrs if a['cutoff']==2025 and a['question_id']!=q['question_id'] and a['items'] and a['items'][0]['text']!=e['text']),None)
    if alternative: records.append(dict(kind='mismatched_premise_control',question_id=q['question_id'],premise=alternative['items'][0]['text'],hypothesis=req['hypothesis']))
nli=ControlledLocalNLI(read_config('configs/constraints.yaml'))
probs,cost=nli.predict_controlled([(r['premise'],r['hypothesis']) for r in records])
for r,p in zip(records,probs): r['nli']=p
interpretation='These post-hoc controls do not enter any ranking. Exact identity/negation and paired comparison subjects test logical consistency; numerical extrapolation is not entailed merely by a smaller-system result. Mismatched graph premises are an imperfect negative control, not independently annotated false statements. The probabilities and any counterexamples are retained even when they undermine the NLI model.'
write_json(root/'constraints'/'nli_control_probes.json',dict(model=nli.metadata,source_hash=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),cost=cost,records=records,interpretation=interpretation))
for r in records:
    print(r['kind'],r.get('question_id',r.get('node_id','')),round(r['nli']['entailment'],4),round(r['nli']['neutral'],4),round(r['nli']['contradiction'],4),flush=True)
