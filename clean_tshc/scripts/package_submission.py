"""Package only the clean submission and accepted outputs, never the old stack."""
import argparse
import hashlib
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tkh_abstraction.data import read,write,sha


def package(output,archive):
    root=Path(__file__).resolve().parents[1]
    accepted=read(root/output/'audit/acceptance.json')
    gate=read(root/output/'audit/gate_K.json')
    assert accepted['status']=='PASS' and gate['status']=='PASS'
    selected=[]
    for name in ['README.md','report.md','AI_USAGE.md','requirements.txt','references.md']:
        selected.append((root/name,name))
    for folder in ['src','scripts','tests','configs','data','notes']:
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pid'] and p.name not in ['package_manifest.json','package_validation.json']:
                selected.append((p,p.relative_to(root).as_posix()))
    for p in sorted((root/output).rglob('*')):
        if p.is_file():selected.append((p,'outputs/'+p.relative_to(root/output).as_posix()))
    manifest={name:sha(p) for p,name in selected}
    write(root/'notes/package_manifest.json',manifest)
    with zipfile.ZipFile(root/archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,name in selected:z.write(p,name)
        z.write(root/'notes/package_manifest.json','package_manifest.json')
    with zipfile.ZipFile(root/archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in manifest.items())
    result=dict(archive=archive,bytes=(root/archive).stat().st_size,sha256=sha(root/archive),files=len(selected)+1,accepted_output=output,crc_and_all_file_hashes_verified=True)
    write(root/'notes/package_validation.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='outputs');ap.add_argument('--archive',default='tshc_submission.zip');args=ap.parse_args();print(package(args.output,args.archive))
