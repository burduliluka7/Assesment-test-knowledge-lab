"""Pinned, local NLI model shared by independent support checks."""
import numpy as np
from pathlib import Path
from .io import read_json,write_json,digest

class NLI:
    def __init__(self,model,revision):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(model,revision=revision,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(model,revision=revision,use_safetensors=True,local_files_only=True).eval()
        self.labels={int(k):str(v).lower() for k,v in self.model.config.id2label.items()}
        if set(self.labels.values())!={'entailment','neutral','contradiction'}: raise ValueError('Explicit NLI class names required')
        self.metadata=dict(model=model,revision=revision,threshold_unchanged=.5,max_length=512)
        self.cache_path=Path('.cache/nli')/(digest(self.metadata)+'.json')
        self.cache=read_json(self.cache_path) if self.cache_path.exists() else {}

    def predict(self,pairs,batch_size=12):
        keys=[digest([a,b]) for a,b in pairs]; missing={k:p for k,p in zip(keys,pairs) if k not in self.cache}
        if missing:
            values=self._predict_uncached(list(missing.values()),batch_size)
            self.cache.update(dict(zip(missing,values))); write_json(self.cache_path,self.cache)
        return [self.cache[k] for k in keys]

    def _predict_uncached(self,pairs,batch_size=12):
        import torch
        rows=[]
        for start in range(0,len(pairs),batch_size):
            batch=pairs[start:start+batch_size]
            lengths=[len(self.tokenizer(a,b)['input_ids']) for a,b in batch]
            inputs=self.tokenizer([a for a,b in batch],[b for a,b in batch],padding=True,truncation='only_first',max_length=512,return_tensors='pt')
            with torch.inference_mode(): values=self.model(**inputs).logits.softmax(-1).numpy()
            for p,length in zip(values,lengths):
                row={self.labels[i]:float(x) for i,x in enumerate(p)}
                row.update(prediction=self.labels[int(np.argmax(p))],truncated=length>512,input_tokens=length)
                rows.append(row)
        return rows
