"""Pin the installed transitive dependencies actually required by this project."""
from importlib.metadata import distribution
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pathlib import Path
roots=['numpy','scipy','scikit-learn','PyYAML','matplotlib','sentence-transformers','torch','transformers','pytest','setuptools']
pending=roots.copy(); seen={}
while pending:
    name=canonicalize_name(pending.pop())
    if name in seen: continue
    d=distribution(name); seen[name]=d.version
    for text in d.requires or []:
        req=Requirement(text)
        if req.marker is None or req.marker.evaluate({'extra':''}): pending.append(req.name)
Path('requirements.txt').write_text('# Python 3.11; versions from the verified execution environment.\n'+'\n'.join(f'{n}=={v}' for n,v in sorted(seen.items()))+'\n',encoding='utf-8')
print(f'Pinned {len(seen)} relevant distributions')
