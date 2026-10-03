"""Persistent local process wrapper with UTF8 logs; predictor still runs separately."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
role=sys.argv[1] if len(sys.argv)>1 else 'prediction'
logs=ROOT/'artifacts/run_logs';logs.mkdir(parents=True,exist_ok=True)
command=[str(ROOT.parent/'rest_of_work/.repro-venv/Scripts/python.exe'),'-B',str(ROOT/'scripts/predict.py')]
if role=='reproduction':command+=['--output','artifacts/reproduction']
start=time.perf_counter()
with (logs/(role+'_stdout.log')).open('w',encoding='utf8') as out,(logs/(role+'_stderr.log')).open('w',encoding='utf8') as err:
    process=subprocess.Popen(command,stdout=out,stderr=err)
    (logs/(role+'_process.json')).write_text(json.dumps(dict(pid=process.pid,command=command)),encoding='utf8')
    code=process.wait()
(logs/(role+'_exit.json')).write_text(json.dumps(dict(exit_code=code,wall_seconds=time.perf_counter()-start)),encoding='utf8')
