"""Ask the user-authorized Ultralight report author after frozen evaluation exists."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
notes=ROOT/'notes/requirement_design'
while not (notes/'report_handoff.json').exists():time.sleep(5)
with (notes/'report_result.json').open('w',encoding='utf8') as out,(notes/'report_stderr.log').open('w',encoding='utf8') as err:
    result=subprocess.run([sys.executable,r'C:\Users\Luka Burduli\.codex\ultralight\claude_delegate.py','planner','--effort','medium','--cwd',str(ROOT),'--prompt-file',str(notes/'report_prompt.txt')],stdout=out,stderr=err)
(notes/'report_exit.json').write_text(json.dumps(dict(exit_code=result.returncode)),encoding='utf8')
