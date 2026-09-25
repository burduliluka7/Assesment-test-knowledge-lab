"""Build a local submission archive; never uploads or publishes anything."""
from pathlib import Path
import hashlib
import json
import zipfile

root=Path('.').resolve()
paths=set(root.glob('*.md'))|{root/'metrics.json',root/'pyproject.toml',root/'requirements.txt',root/'.gitignore'}
for folder in ['src','tests','configs','data','scripts','report','notes','artifacts/full','artifacts/evaluation','artifacts/baseline_v1','artifacts/baseline_v2','artifacts/baseline_contextual','artifacts/baseline_diffusion','artifacts/baseline_reranking','artifacts/baseline_constraints']:
    paths.update(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo'])
paths.add(root/'artifacts/submission_checklist.md')
manifest={}
with zipfile.ZipFile(root/'submission.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(paths):
        rel=p.relative_to(root).as_posix(); archive.write(p,rel)
        manifest[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    archive.writestr('MANIFEST.json',json.dumps(dict(files=manifest,description='Local assessment submission; SHA-256 hashes of uncompressed files'),indent=2))
with zipfile.ZipFile(root/'submission.zip') as archive:
    assert set(archive.namelist())==set(manifest)|{'MANIFEST.json'}
    for rel,expected in manifest.items():
        assert hashlib.sha256(archive.read(rel)).hexdigest()==expected, f'Archive hash mismatch: {rel}'
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected, f'File changed during packaging: {rel}'
print(f'Submission: {len(manifest)} files, {(root/"submission.zip").stat().st_size/1e6:.1f} MB')
print('Verified every archived SHA-256 against the manifest and current workspace.')
