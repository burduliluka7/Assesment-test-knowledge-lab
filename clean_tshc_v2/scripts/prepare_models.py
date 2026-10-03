"""Populate ONLY V2's model cache. Reads no benchmark data."""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
from tkh_abstraction_v2.io import write, sha


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Explicitly allow downloading the pinned revisions')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config.json').read_text())
    results = {}
    for role in ('dense', 'relevance', 'nli'):
        name, revision = cfg[role + '_model'], cfg[role + '_revision']
        dest = ROOT / '.cache' / 'models' / revision
        old = Path.home() / '.cache' / 'huggingface' / 'hub' / ('models--' + name.replace('/', '--')) / 'snapshots' / revision
        try:
            if old.is_dir() and not dest.exists():
                shutil.copytree(old, dest, symlinks=False)
            elif not (dest / 'config.json').exists() and args.download:
                snapshot_download(name, revision=revision, local_dir=str(dest),
                    allow_patterns=['*.json', '*.txt', '*.safetensors', '1_Pooling/*'],
                    ignore_patterns=['onnx/*', 'openvino/*'])
            elif not (dest / 'config.json').exists():
                raise FileNotFoundError('Pinned local model absent; rerun with --download to explicitly permit network access')
            results[role] = dict(status='available', model=name, revision=revision, path=str(dest.relative_to(ROOT)),
                files={str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and '.cache' not in p.relative_to(dest).parts})
        except Exception as exc:
            results[role] = dict(status='unavailable', model=name, revision=revision, reason=repr(exc))
        print(role, results[role]['status'], flush=True)
        write(ROOT / 'model_manifest.json', results)


if __name__ == '__main__':
    main()
