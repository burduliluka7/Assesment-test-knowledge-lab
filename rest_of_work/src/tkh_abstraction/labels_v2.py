"""Generation is isolated from deterministic held-out member evidence."""
from collections import Counter,defaultdict
import csv, hashlib, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from .io import digest

BANNED={'best','leading','superior','universal','first','outperforming','outperforms','novel','state-of-the-art'}

def overclaim(text):
    return bool(re.search(r'\b(?:best|leading|superior|universal|first|novel|outperform\w*|sota|state[ -]of[ -]the[ -]art)\b',text,re.I))

def evidence_split(nodes,seed=42,fraction=.3):
    if len(nodes)<6: return list(nodes),[]
    grouped=defaultdict(list)
    for n in nodes: grouped[n['type']].append(n)
    generation=[]; verification=[]
    for typ,group in sorted(grouped.items()):
        ordered=sorted(group,key=lambda n:hashlib.sha256(f'{seed}:{n["id"]}'.encode()).hexdigest())
        count=min(len(group)-1,max(1,round(fraction*len(group)))) if len(group)>1 else 0
        verification.extend(ordered[:count]); generation.extend(ordered[count:])
    if len(verification)<2 or len(generation)<2: return list(nodes),[]
    return generation,verification

def evidence_row(n):
    return dict(id=n['id'],type=n['type'],surface_form=n['surface_form'],first_seen_year=n['first_seen_year'])

def generation_packet(nodes,snap):
    ids={n['id'] for n in nodes}; counts=Counter(n['type'] for n in nodes)
    relations=Counter(e['relation_type'] for e in snap['hyperedges'] if set(e['members'])<=ids)
    # No global snapshot IDF and no verification text in any generator feature.
    return dict(cutoff=snap['snapshot'],generation_members=[evidence_row(n) for n in nodes],
        generation_type_counts=dict(counts),generation_only_relation_counts=dict(relations))

def generate_from_packet(packet):
    members=packet['generation_members']; texts=[n['surface_form'] for n in members]
    normalized_texts=[re.sub(r'[^\w]+',' ',text.casefold()) for text in texts]
    vectorizer=TfidfVectorizer(ngram_range=(1,3),lowercase=True,max_features=15000)
    try:
        T=vectorizer.fit_transform(texts); features=vectorizer.get_feature_names_out()
        means=np.asarray(T.mean(axis=0)).ravel(); frequencies=np.asarray((T>0).sum(axis=0)).ravel()
        options=[]
        for i,phrase in enumerate(features):
            words=phrase.split()
            if len(words)<2 or any(w in ENGLISH_STOP_WORDS or w in BANNED for w in words) or overclaim(phrase): continue
            if re.search(r'\d',phrase): continue
            if not any(phrase in text for text in normalized_texts): continue
            options.append((float(means[i]*np.sqrt(frequencies[i])*len(words)),str(phrase)))
        phrase=max(options,key=lambda p:(p[0],p[1]))[1] if options else None
    except ValueError: phrase=None
    if not phrase:
        # A literal short member name is more defensible than invented abstraction.
        candidates=[t.strip().rstrip('.') for t in texts if len(t.split())<=8 and not overclaim(t)]
        phrase=min(candidates,key=lambda t:(len(t),t)) if candidates else 'Scientific corpus entries'
    label=phrase[0].upper()+phrase[1:]
    gloss=f'One topic represented by these corpus entries is {phrase}.'
    return label,gloss

def label_overlay(h,snap,cfg):
    lookup={n['id']:n for n in snap['nodes']}; logs=[]; overlays=[]
    for k,level in enumerate(h['levels']):
        for c in level:
            members=[lookup[x] for x in c['member_ids']]
            if any(n['first_seen_year']>snap['snapshot'] for n in members): raise ValueError('Future label evidence')
            if len(members)==1:
                overlays.append(dict(id=c['id'],snapshot=snap['snapshot'],level=k,label=c['label'],gloss=c['gloss'])); continue
            gen,held=evidence_split(members,cfg['seed'],cfg['v2_label_holdout_fraction'])
            packet=generation_packet(gen,snap); label,gloss=generate_from_packet(packet)
            overlays.append(dict(id=c['id'],snapshot=snap['snapshot'],level=k,label=label,gloss=gloss))
            logs.append(dict(id=c['id'],snapshot=snap['snapshot'],level=k,generator='extractive-heldout-v2',
                generation_input=packet,verification_evidence=[evidence_row(n) for n in held],
                holdout_status='available' if held else 'not_applicable_small_or_type_sparse_cluster',
                label=label,gloss=gloss,legacy_label=c['label'],legacy_gloss=c['gloss']))
    return overlays,logs

def label_tokens(label):
    return set(re.findall(r'\w+',label.casefold()))-ENGLISH_STOP_WORDS

def verification_chunks(log,max_members=6):
    evidence=log['verification_evidence']
    ordered=sorted(evidence,key=lambda n:hashlib.sha256(n['id'].encode()).hexdigest())[:max_members]
    # Fixed-size chunks reduce lost evidence; actual token truncation is recorded by NLI.
    chunks=[]
    for i in range(0,len(ordered),2):
        group=ordered[i:i+2]
        chunks.append(dict(member_ids=[n['id'] for n in group],premise='Verification corpus entries: '+' '.join(n['surface_form'] for n in group)))
    return chunks

def evaluate_labels(logs,nli,cfg,outdir):
    rng=np.random.default_rng(cfg['seed']); sample=[]
    for y in sorted({x['snapshot'] for x in logs}):
        for level in [0,1]:
            group=sorted([x for x in logs if x['snapshot']==y and x['level']==level],key=lambda x:x['id'])
            sample.extend(group[i] for i in rng.permutation(len(group))[:cfg['v2_label_sample_per_level']])
    pairs=[]; work=[]
    for log in sample:
        chunks=verification_chunks(log)
        for version in ['legacy','improved']:
            gloss=log['legacy_gloss'] if version=='legacy' else log['gloss']
            begin=len(pairs); pairs.extend((c['premise'],gloss) for c in chunks)
            work.append((log,version,chunks,begin,len(pairs)))
    pred=nli.predict(pairs); rows=[]
    for log,version,chunks,a,b in work:
        label=log['legacy_label'] if version=='legacy' else log['label']; probs=pred[a:b]
        supported=label_tokens(label)<=set().union(*(label_tokens(n['surface_form']) for n in log['verification_evidence'])) if log['verification_evidence'] else None
        # An existential gloss needs at least one supporting held-out chunk. No cherry-picked chunk selection.
        best=max(probs,key=lambda p:p['entailment']) if probs else None
        rows.append(dict(cluster_id=log['id'],snapshot=log['snapshot'],level=log['level'],version=version,
            label=label,gloss=log['legacy_gloss'] if version=='legacy' else log['gloss'],
            holdout_status=log['holdout_status'],label_lexically_supported=supported,
            nli_status='computed' if best else 'not_applicable',nli=best,
            unsupported=best['entailment']<cfg['nli_threshold'] if best else None,
            chunks=[dict(**c,nli=p) for c,p in zip(chunks,probs)],
            generation_members=len(log['generation_input']['generation_members']),verification_members=len(log['verification_evidence'])))
        held=log['verification_evidence']; lt=label_tokens(label); union=set().union(*(label_tokens(n['surface_form']) for n in held))
        rows[-1]['label_token_coverage']=len(lt&union)/len(lt) if held and lt else None
        rows[-1]['heldout_member_topic_coverage']=sum(bool(lt&label_tokens(n['surface_form'])) for n in held)/len(held) if held else None
        rows[-1]['chunk_entailment_rate']=sum(p['entailment']>=cfg['nli_threshold'] for p in probs)/len(probs) if probs else None
    controls=[]; control_pairs=[]
    for log in sample:
        if not log['verification_evidence']: continue
        others=sorted([x for x in sample if x['snapshot']==log['snapshot'] and x['level']==log['level'] and x['id']!=log['id'] and x['verification_evidence']],key=lambda x:x['id'])
        own=verification_chunks(log)[0]['premise']
        gen=dict(log,verification_evidence=log['generation_input']['generation_members'])
        entry=dict(cluster_id=log['id'],snapshot=log['snapshot'],level=log['level'],other_cluster_id=others[0]['id'] if others else None)
        for kind,premise in [('own_holdout',own),('generation',verification_chunks(gen)[0]['premise']),('other_cluster',verification_chunks(others[0])[0]['premise'] if others else None)]:
            if premise is not None:
                controls.append(dict(**entry,kind=kind,premise=premise)); control_pairs.append((premise,log['gloss']))
    for row,p in zip(controls,nli.predict(control_pairs)): row['nli']=p
    summaries={}
    for version in ['legacy','improved']:
        summaries[version]={}
        for level in [0,1]:
            rr=[r for r in rows if r['version']==version and r['level']==level]; eligible=[r for r in rr if r['nli_status']=='computed']
            summaries[version][str(level)]=dict(status='computed' if eligible else 'not_applicable',sampled=len(rr),evaluable=len(eligible),
                holdout_coverage=len(eligible)/len(rr) if rr else None,
                lexical_label_support_rate=float(np.mean([r['label_lexically_supported'] for r in eligible])) if eligible else None,
                unsupported_rate=float(np.mean([r['unsupported'] for r in eligible])) if eligible else None,
                prediction_rates={name:sum(r['nli']['prediction']==name for r in eligible)/len(eligible) if eligible else None for name in ['entailment','neutral','contradiction']},
                truncated_chunks=sum(p['nli']['truncated'] for r in eligible for p in r['chunks']))
            for metric in ['label_token_coverage','heldout_member_topic_coverage','chunk_entailment_rate']:
                summaries[version][str(level)][metric]=float(np.mean([r[metric] for r in eligible])) if eligible else None
    outdir.mkdir(parents=True,exist_ok=True); audit=[]; audit_key=[]
    for i in rng.permutation(len(rows)):
        r=rows[i]; audit.append(dict(audit_id=f'V2-{len(audit):04}',label=r['label'],gloss=r['gloss'],evidence=' | '.join(c['premise'] for c in r['chunks']),rating='',notes=''))
        audit_key.append(dict(audit_id=audit[-1]['audit_id'],cluster_id=r['cluster_id'],snapshot=r['snapshot'],level=r['level'],version=r['version']))
    with (outdir/'audit_blind.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['audit_id','label','gloss','evidence','rating','notes']); w.writeheader(); w.writerows(audit)
    from .io import write_json
    write_json(outdir/'audit_key.json',audit_key)
    return dict(status='computed',model=nli.metadata,threshold=cfg['nli_threshold'],
        protocol='Same fixed <=6 held-out members and <=3 chunks for paired old/new glosses; existential support uses maximum entailment, not scientific truth verification. No benchmark text enters generation.',
        summaries=summaries,rows=rows,controls=controls,control_summary={kind:float(np.mean([r['nli']['entailment']>=cfg['nli_threshold'] for r in controls if r['kind']==kind])) for kind in sorted({r['kind'] for r in controls})},human_audit=dict(status='missing_data',reason='Blind audit exported; no human ratings fabricated'))
