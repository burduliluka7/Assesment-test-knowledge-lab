"""Compare both read-only project trees with their user-approved starting state."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_reference():
    notes=ROOT/'notes/requirement_design';out=ROOT/'artifacts/verification';out.mkdir(parents=True,exist_ok=True)
    status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT.parent)
    (out/'git_status_final.txt').write_bytes(status)
    initial=(notes/'git_status_before.txt').read_bytes()
    def entries(data,name):return sorted(line for line in data.decode('utf8',errors='replace').splitlines() if ' '+name+'/' in line)
    results={}
    for name in ('clean_tshc','rest_of_work'):
        before=json.loads((notes/(name+'_before.json')).read_text(encoding='utf8'));folder=ROOT.parent/name
        after={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()}
        diff=subprocess.check_output(['git','diff','--binary','--',name],cwd=ROOT.parent)
        normal=subprocess.check_output(['git','diff','--',name],cwd=ROOT.parent)
        result=dict(unchanged=before==after,files=len(after),git_diff_matches_start=diff==(notes/(name+'_diff_before.patch')).read_bytes(),
            git_status_matches_start=entries(status,name)==entries(initial,name),
            added=sorted(set(after)-set(before)),removed=sorted(set(before)-set(after)),
            changed=[n for n in before if n in after and before[n]!=after[n]])
        (out/(name+'_after.json')).write_text(json.dumps(after,indent=2),encoding='utf8')
        (out/(name+'_diff_final.patch')).write_bytes(diff);(out/(name+'_git_diff.txt')).write_bytes(normal)
        results[name]=result
        print(name, result['files'], 'files; unchanged:',result['unchanged'],flush=True)
    result=dict(trees=results,unchanged=all(v['unchanged'] and v['git_diff_matches_start'] and v['git_status_matches_start'] for v in results.values()),
        checked_at=datetime.now(timezone.utc).isoformat(),interpretation='Zero additional protected-tree changes. Pre-existing user-approved V1 report changes retained.')
    (out/'protected_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    assert result['unchanged'], 'Protected tree changed from initial state'
    return result


if __name__=='__main__':print(json.dumps(verify_reference(),indent=2))
