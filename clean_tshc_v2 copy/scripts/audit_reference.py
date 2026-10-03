"""Preserve the user-approved starting V1 bytes AND its pre-existing git diff."""
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_reference():
    notes=ROOT/'notes/entity_redesign';out=ROOT/'outputs/audit';out.mkdir(parents=True,exist_ok=True)
    before=json.loads((notes/'v1_before.json').read_text(encoding='utf8'))
    protected=ROOT.parent/'clean_tshc'
    after={p.relative_to(protected).as_posix():digest(p) for p in protected.rglob('*') if p.is_file()}
    diff=subprocess.check_output(['git','diff','--binary','--','clean_tshc'],cwd=ROOT.parent)
    standard_diff=subprocess.check_output(['git','diff','--','clean_tshc'],cwd=ROOT.parent)
    status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT.parent)
    original_diff=(notes/'v1_diff_before.patch').read_bytes()
    def v1_status(data):return sorted(line for line in data.decode('utf8',errors='replace').splitlines() if ' clean_tshc/' in line)
    original_status=v1_status((notes/'git_status_before.txt').read_bytes())
    result=dict(unchanged=before==after,files=len(after),git_diff_matches_start=diff==original_diff,
        git_status_matches_start=v1_status(status)==original_status,preexisting_v1_status=original_status,
        added=sorted(set(after)-set(before)),removed=sorted(set(before)-set(after)),
        changed=[n for n in before if n in after and before[n]!=after[n]],
        checked_at=datetime.now(timezone.utc).isoformat(),
        interpretation='ZERO NEW V1 modifications; existing user-approved report changes preserved. Git diff was not initially empty.')
    (out/'v1_after.json').write_text(json.dumps(after,indent=2),encoding='utf8')
    (out/'v1_diff_final.patch').write_bytes(diff)
    (out/'git_diff_clean_tshc.txt').write_bytes(standard_diff)
    (out/'git_status_final.txt').write_bytes(status)
    (out/'protected_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    assert result['unchanged'] and result['git_diff_matches_start'] and result['git_status_matches_start'], 'V1 differs from task-start baseline; investigate before finishing'
    return result


if __name__=='__main__':print(json.dumps(verify_reference(),indent=2))
