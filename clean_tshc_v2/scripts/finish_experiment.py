"""Post-freeze local workflow; no predictive policy changes or result-driven tuning."""
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT.parent/'rest_of_work/.repro-venv/Scripts/python.exe'
LOGS=ROOT/'artifacts/run_logs'
AUDIT=ROOT/'artifacts/verification';AUDIT.mkdir(parents=True,exist_ok=True)
freeze=ROOT/'artifacts/prediction/audit/freeze.json'
while not freeze.exists():time.sleep(5)
# Predictor emits its final event just after the manifest; wait until it exits.
while True:
    info=json.loads((LOGS/'prediction_exit.json').read_text())
    if info['exit_code']==0:break
    time.sleep(2)

def run(name,args,folder=LOGS):
    start=time.perf_counter()
    with (folder/(name+'_stdout.log')).open('w',encoding='utf8') as out,(folder/(name+'_stderr.log')).open('w',encoding='utf8') as err:
        result=subprocess.run([str(PYTHON),'-B',*args],cwd=ROOT.parent,stdout=out,stderr=err)
    (folder/(name+'_exit.json')).write_text(json.dumps(dict(exit_code=result.returncode,wall_seconds=time.perf_counter()-start)),encoding='utf8')
    if result.returncode:raise SystemExit(name+' failed')

run('evaluation',[str(ROOT/'scripts/evaluate.py')])
run('materials',[str(ROOT/'scripts/build_report_materials.py')])
# Independent prediction process, same source/config and source-keyed model caches.
replay=subprocess.Popen([str(PYTHON),'-B',str(ROOT/'scripts/run_local.py'),'reproduction'],cwd=ROOT.parent)
replay.wait()
assert json.loads((LOGS/'reproduction_exit.json').read_text())['exit_code']==0
start=time.perf_counter()
with (AUDIT/'tests_final.log').open('w',encoding='utf8') as out,(AUDIT/'tests_final_stderr.log').open('w',encoding='utf8') as err:
    test=subprocess.run([str(PYTHON),'-B','-m','pytest',str(ROOT/'tests'),'-q','-p','no:cacheprovider'],cwd=ROOT.parent,stdout=out,stderr=err)
(AUDIT/'tests_exit.json').write_text(json.dumps(dict(exit_code=test.returncode,wall_seconds=time.perf_counter()-start)),encoding='utf8')
(LOGS/'workflow_completed.json').write_text(json.dumps(dict(tests_exit_code=test.returncode)),encoding='utf8')
