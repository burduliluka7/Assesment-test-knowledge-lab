"""Gold-blind active V2 entry point. No scoring-data imports."""
import argparse
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
for key in ('HF_HOME','TORCH_HOME','MPLCONFIGDIR','TMP','TEMP'):
    folder=ROOT/'.cache'/key.lower()
    folder.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(folder)
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
from tkh_abstraction_v2.isolation import install
accesses=install(ROOT,prediction=True)
from tkh_abstraction_v2.entity_experiment import run
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='artifacts/prediction')
    args=parser.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must stay inside V2')
    run(ROOT,out,accesses)
