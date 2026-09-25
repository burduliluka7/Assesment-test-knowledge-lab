"""Download only configured public model weights; no dataset uploads."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tkh_abstraction.config import read_config
from tkh_abstraction.semantics import Encoder
from tkh_abstraction.io import write_json
cfg=read_config('configs/default.yaml')
for purpose in ['embedding','evaluation']:
    enc=Encoder(cfg[f'{purpose}_model'],cfg[f'{purpose}_revision'])
    X=enc.encode(['A scientific method for atomic potentials.'],'model-smoke-test')
    print(purpose,enc.metadata,X.shape,flush=True)
from transformers import AutoTokenizer,AutoModelForSequenceClassification
name=cfg['nli_model']; revision=cfg['nli_revision']
tok=AutoTokenizer.from_pretrained(name,revision=revision)
model=AutoModelForSequenceClassification.from_pretrained(name,revision=revision,use_safetensors=True)
print('nli',model.config.id2label,flush=True)
