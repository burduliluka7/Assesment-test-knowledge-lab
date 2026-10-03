"""Compile both V2 reports locally, without package downloads or shell escape."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]


def main():
    audit=ROOT/'artifacts/verification';audit.mkdir(parents=True,exist_ok=True)
    executable=shutil.which('pdflatex')
    if not executable:raise RuntimeError('Install a local LaTeX distribution with pdflatex first')
    environment=dict(os.environ)
    environment['PATH']=os.pathsep.join(p for p in environment.get('PATH','').split(os.pathsep) if Path(p).is_dir())
    reports={}
    for stem in ('v2_entity_retrieval_report','v2_assessment_companion'):
        report=ROOT/'report'/(stem+'.tex');codes=[]
        for number in (1,2):
            with (audit/f'{stem}_pass{number}.log').open('w',encoding='utf8') as log:
                result=subprocess.run([executable,'-disable-installer','-no-shell-escape','-interaction=nonstopmode','-halt-on-error',report.name],
                    cwd=report.parent,env=environment,stdout=log,stderr=subprocess.STDOUT)
            codes.append(result.returncode)
            if result.returncode:break
        reports[stem]=dict(exit_codes=codes,tex_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
        print(stem,'LaTeX exit codes:',codes)
    result=dict(reports=reports,**reports['v2_entity_retrieval_report'])
    (audit/'report_compile.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    if any(r['exit_codes']!=[0,0] for r in reports.values()):raise SystemExit(1)


if __name__=='__main__':main()
