"""Compile the supplied Claude-authored report locally, without package downloads."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT / 'report/v2_entity_retrieval_report.tex'
    audit = ROOT / 'outputs/audit'
    executable = shutil.which('pdflatex')
    if not executable:
        raise RuntimeError('Install a local LaTeX distribution with pdflatex first')
    environment = dict(os.environ)
    # Some IDE launchers include an executable (bash.exe) as a PATH directory.
    # Remove invalid entries for this subprocess only; do not change user settings.
    environment['PATH'] = os.pathsep.join(p for p in environment.get('PATH', '').split(os.pathsep) if Path(p).is_dir())
    codes = []
    for number in (1, 2):
        with (audit / f'latex_pass{number}.log').open('w', encoding='utf8') as log:
            result = subprocess.run([executable, '-disable-installer', '-no-shell-escape',
                '-interaction=nonstopmode', '-halt-on-error', report.name], cwd=report.parent,
                env=environment, stdout=log, stderr=subprocess.STDOUT)
        codes.append(result.returncode)
        if result.returncode:
            break
    (audit / 'report_compile.json').write_text(json.dumps(dict(exit_codes=codes,
        tex_sha256=hashlib.sha256(report.read_bytes()).hexdigest()), indent=2), encoding='utf8')
    print('LaTeX exit codes:', codes)
    if codes != [0, 0]:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
