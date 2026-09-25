"""Ground-truth alignment only. This module is never imported by ranking code."""
from collections import defaultdict
import re,unicodedata
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from ..target_mapping import normalize

def tokens(text):
    return set(re.findall(r'\w+',normalize(text)))-ENGLISH_STOP_WORDS

def lexical_overlap(a,b):
    aa,bb=tokens(a),tokens(b)
    return 2*len(aa&bb)/(len(aa)+len(bb)) if aa or bb else 0.

def plain(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(text)).casefold() if not unicodedata.combining(c))

class SourceIndex:
    def __init__(self,snap):
        nodes={n['id']:n for n in snap['nodes']}; self.articles={}; self.authors=defaultdict(set)
        for n in nodes.values():
            if n['type']=='article':
                for aid in n.get('provenance',{}).get('articles',[]): self.articles[aid]=n
        for e in snap['hyperedges']:
            if e['relation_type']=='authored_by':
                aid=e.get('provenance',{}).get('article_id')
                self.authors[aid].update(re.findall(r'\w+',plain(nodes[x]['surface_form']))[-1] for x in e['members'] if nodes[x]['type']=='author')

    def resolve(self,source):
        if isinstance(source,str): source={'key':source}
        title=source.get('title',''); doi=(source.get('doi') or '').casefold().removeprefix('https://doi.org/')
        matches=[]; rule=None
        for aid,n in self.articles.items():
            nd=(n.get('provenance',{}).get('doi') or '').casefold().removeprefix('https://doi.org/')
            if (doi and nd==doi) or (title and normalize(title)==normalize(n['surface_form'])):
                matches.append(aid); rule='doi_or_exact_title'
        if not matches:
            ref=source.get('key',''); years=re.findall(r'\b(?:19|20)\d{2}\b',ref+' '+source.get('date',''))
            surname=re.findall(r'\w+',plain(ref))
            if years and surname:
                for aid,n in self.articles.items():
                    if n['first_seen_year']==int(years[0]) and surname[0] in self.authors[aid]: matches.append(aid)
                rule='author_surname_and_year_candidate'
        return dict(source=source,article_ids=sorted(matches),rule=rule,
            status='matched' if len(matches)==1 else 'ambiguous' if matches else 'unmapped')

def benchmark_claims(gt):
    items=[]
    for qid,g in gt.items():
        if g['type']=='A':
            for entity,claims in g.get('method_claims',{}).items():
                evidence=g.get('method_evidence',{}).get(entity,[])
                refs=[e['source'] for e in evidence if e.get('source')]
                if not refs: refs=g.get('required_citations',[])
                for j,text in enumerate(claims):
                    items.append(dict(item_id=f'{qid}:{entity}:claim:{j}',question_id=qid,question_type='A',kind='claim',target_entity=entity,
                        ground_truth_claim=text,sources=refs,evidence=evidence))
            for entity,evidence in g.get('method_evidence',{}).items():
                for j,e in enumerate(evidence):
                    items.append(dict(item_id=f'{qid}:{entity}:evidence:{j}',question_id=qid,question_type='A',kind='evidence',target_entity=entity,
                        ground_truth_claim=e.get('evidence',''),sources=[e['source']] if e.get('source') else [],evidence=e))
        else:
            for c in g.get('required_claims',[]):
                refs=[g.get('sources',{}).get(ref,{'key':ref}) for ref in c.get('sources',[])]
                items.append(dict(item_id=c['id'],question_id=qid,question_type='B',kind='claim',target_entity=None,
                    ground_truth_claim=c['text'],sources=refs,evidence=c.get('evidence',''),
                    benchmark_coverage=c.get('coverage'),caveat=c.get('caveat'),flag=c.get('flag')))
    return items

def decide_alignment(candidates,cfg):
    credible=[c for c in candidates if c['source_supported'] and c['cosine']>=cfg['v2_alignment_cosine']
        and c['lexical_overlap']>=cfg['v2_alignment_lexical'] and c.get('nli',{}).get('entailment',0)>=cfg['v2_alignment_entailment']
        and not c.get('nli',{}).get('truncated',True) and c['numeric_support']]
    if not credible: return 'unmapped',[]
    credible.sort(key=lambda c:(-c['cosine'],c['node_id']))
    if len(credible)>1 and credible[0]['cosine']-credible[1]['cosine']<cfg['v2_alignment_margin'] and normalize(credible[0]['surface_form'])!=normalize(credible[1]['surface_form']):
        return 'ambiguous',[]
    text=normalize(credible[0]['surface_form'])
    return 'confident_proxy',[c['node_id'] for c in credible if normalize(c['surface_form'])==text]

def align(gt,snap,encoder,nli,cfg):
    source_index=SourceIndex(snap); claims=[n for n in snap['nodes'] if n['type']=='claim']; items=benchmark_claims(gt)
    C=encoder.encode([n['surface_form'] for n in claims],f'claim-alignment-corpus:{snap["snapshot"]}')
    T=encoder.encode([x['ground_truth_claim'] for x in items],f'claim-alignment-benchmark:{snap["snapshot"]}')
    rows=[]; pairs=[]; locations=[]
    for item,v in zip(items,T):
        refs=[source_index.resolve(s) for s in item['sources']]
        source_ids={a for r in refs for a in r['article_ids']}; definite={a for r in refs if r['status']=='matched' for a in r['article_ids']}
        pool=[i for i,n in enumerate(claims) if source_ids&set(n.get('provenance',{}).get('articles',[]))]
        if not pool: pool=list(range(len(claims))) # diagnostics only; never accepted without a matched source
        nearest=sorted(pool,key=lambda i:(-float(C[i]@v),claims[i]['id']))[:3]
        candidates=[]
        for i in nearest:
            n=claims[i]; numbers=set(re.findall(r'\b\d+(?:\.\d+)?\b',item['ground_truth_claim']))
            candidate=dict(node_id=n['id'],surface_form=n['surface_form'],provenance=n.get('provenance',{}),
                cosine=float(C[i]@v),lexical_overlap=lexical_overlap(item['ground_truth_claim'],n['surface_form']),
                source_supported=bool(definite&set(n.get('provenance',{}).get('articles',[]))),
                numeric_support=numbers<=set(re.findall(r'\b\d+(?:\.\d+)?\b',n['surface_form'])))
            if candidate['source_supported'] and candidate['cosine']>=cfg['v2_alignment_cosine'] and candidate['lexical_overlap']>=cfg['v2_alignment_lexical']:
                locations.append((len(rows),len(candidates))); pairs.append((n['surface_form'],item['ground_truth_claim']))
            candidates.append(candidate)
        rows.append(dict(**item,cutoff=snap['snapshot'],source_alignment=refs,source_article_ids=sorted(definite),
            source_candidate_count=len(source_ids),candidate_pool_size=len(pool),candidates=candidates))
    predictions=nli.predict(pairs) if pairs else []
    for (i,j),p in zip(locations,predictions): rows[i]['candidates'][j]['nli']=p
    for row in rows:
        row['alignment_status'],row['node_ids']=decide_alignment(row['candidates'],cfg)
        row['interpretation']='Conservative automated alignment proxy; not human-verified scientific truth'
    return dict(status='computed',cutoff=snap['snapshot'],alignment_model=encoder.metadata,nli_model=nli.metadata,
        thresholds={k:cfg[k] for k in cfg if k.startswith('v2_alignment_')},items=rows)

def coverage(alignment):
    result={}
    for typ in ['A','B']:
        rows=[r for r in alignment['items'] if r['question_type']==typ and r['kind']=='claim']
        result[typ]=dict(total=len(rows),confident_proxy=sum(r['alignment_status']=='confident_proxy' for r in rows),
            ambiguous=sum(r['alignment_status']=='ambiguous' for r in rows),unmapped=sum(r['alignment_status']=='unmapped' for r in rows),
            with_matched_source=sum(bool(r['source_article_ids']) for r in rows))
    evidence=[r for r in alignment['items'] if r['kind']=='evidence']
    result['evidence_items']=dict(total=len(evidence),confident_proxy=sum(r['alignment_status']=='confident_proxy' for r in evidence),ambiguous=sum(r['alignment_status']=='ambiguous' for r in evidence),unmapped=sum(r['alignment_status']=='unmapped' for r in evidence))
    return result
