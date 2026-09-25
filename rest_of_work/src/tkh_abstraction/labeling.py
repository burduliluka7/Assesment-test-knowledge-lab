import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .io import digest

def label(h,snap,X):
    """Deterministic keyphrase abstraction, using only the supplied snapshot view."""
    nodes=snap['nodes']; lookup={n['id']:n for n in nodes}; idx={n['id']:i for i,n in enumerate(nodes)}
    texts=[n['surface_form'] for n in nodes]
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=1,max_features=20000)
    try:
        T=vectorizer.fit_transform(texts); names=vectorizer.get_feature_names_out()
    except ValueError: T=None; names=[]
    logs=[]
    for k,level in enumerate(h['levels']):
        for c in level:
            members=[lookup[x] for x in c['member_ids']]
            if any(n['first_seen_year']>snap['snapshot'] for n in members): raise ValueError('future label evidence')
            if len(members)==1:
                c['label']=members[0]['surface_form']; c['gloss']=f"Corpus entry: {members[0]['surface_form'].rstrip('.')} ."
                continue
            ii=[idx[x] for x in c['member_ids']]; values=np.asarray(T[ii].mean(axis=0)).ravel() if T is not None else []
            top=sorted(range(len(values)),key=lambda i:(-values[i],names[i]))
            phrases=[]
            for j in top:
                term=str(names[j])
                if values[j]<=0: break
                if not any(term in p or p in term for p in phrases): phrases.append(term)
                if len(phrases)==3: break
            if not phrases: phrases=['scientific literature']
            c['label']=' / '.join(phrases)
            c['gloss']='This cluster groups corpus entries concerning '+', '.join(phrases)+'.'
            mu=X[ii].mean(axis=0); representatives=sorted(ii,key=lambda i:(-float(X[i]@mu),nodes[i]['id']))[:6]
            # Log ALL generator-visible text; representatives are only a display convenience.
            logs.append(dict(id=c['id'],snapshot=snap['snapshot'],level=k,label=c['label'],gloss=c['gloss'],
                generator='tfidf-keyphrase-template-v1',input=[dict(id=n['id'],type=n['type'],surface_form=n['surface_form'],first_seen_year=n['first_seen_year']) for n in members],
                shared_tfidf_context=dict(snapshot_corpus_file=f'label_corpus_{snap["snapshot"]}.json',
                    ordered_surface_forms_sha256=digest(texts),document_count=len(texts),stop_words='english',ngram_range=[1,2],max_features=20000),
                representatives=[nodes[i]['id'] for i in representatives],keywords=phrases))
    return logs
