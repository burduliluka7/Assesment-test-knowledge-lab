from pathlib import Path
import numpy as np
from .io import digest,write_json,read_json

class Encoder:
    def __init__(self,model,revision='main',cache='.cache/embeddings'):
        self.name=model; self.revision=revision; self.cache=Path(cache); self.model=None
        self.metadata=dict(model=model,requested_revision=revision)

    def load(self):
        if self.model is None:
            if self.name=='synthetic-hash':
                self.metadata['resolved_revision']='sha256-token-hash-v1'; return
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4); torch.manual_seed(0)
            self.model=SentenceTransformer(self.name,revision=self.revision,device='cpu')
            self.model.eval()
            self.metadata['resolved_revision']=getattr(self.model[0].auto_model.config,'_commit_hash',None)

    def encode(self,texts,split):
        self.load()
        key=digest(dict(metadata=self.metadata,texts=texts,split=split))
        path=self.cache/f'{key}.npz'
        if path.exists():
            with np.load(path) as saved: return np.asarray(saved['embeddings'],dtype=np.float32)
        if self.name=='synthetic-hash':
            import re,hashlib
            X=np.zeros((len(texts),64))
            for i,t in enumerate(texts):
                for token in re.findall(r'\w+',t.lower()):
                    j=int(hashlib.sha256(token.encode()).hexdigest()[:8],16)%64; X[i,j]+=1
            X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
        else:
            X=self.model.encode(texts,batch_size=64,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
        X=np.asarray(X,dtype=np.float32)
        self.cache.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,embeddings=X)
        write_json(path.with_suffix('.json'),dict(metadata=self.metadata,split=split,text_hash=digest(texts)))
        return X

def node_texts(nodes,include_type=True):
    return [f"{n['type']}: {n['surface_form']}" if include_type else n['surface_form'] for n in nodes]
