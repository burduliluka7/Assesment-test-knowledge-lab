"""Verify immutable artifacts without importing any gold-aware module."""
import json
from .io import read, sha


def verify_freeze(root, output=None):
    root=root.resolve();output=output or root/'artifacts/prediction'
    manifest=read(output/'audit/freeze.json')
    if manifest.get('status')!='PREDICTIONS_FROZEN_BEFORE_EVALUATION':
        raise ValueError('Incomplete prediction freeze')
    if sha(output/'retrieval/predictions.json')!=manifest['predictions_sha256']:
        raise ValueError('Prediction hash does not match freeze')
    for relative,digest in manifest['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or sha(path)!=digest:
            raise ValueError('Frozen artifact changed: '+relative)
    events=[json.loads(line) for line in (output/'audit/events.jsonl').read_text().splitlines()]
    if not any(e['event']=='predictions_frozen' and e['sha256']==manifest['predictions_sha256'] for e in events):
        raise ValueError('Missing prediction freeze event')
    return manifest


def protect_evaluation(root, output=None):
    from .isolation import install
    output=output or root/'artifacts/prediction'
    manifest=verify_freeze(root,output)
    install(root,prediction=False,frozen_files=[root/p for p in manifest['files']]+[output/'audit/freeze.json',output/'retrieval/predictions.sha256'])
    return manifest
