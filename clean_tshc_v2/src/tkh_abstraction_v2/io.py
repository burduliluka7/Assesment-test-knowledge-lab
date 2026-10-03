import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def event(path, kind, **details):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as stream:
        stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), event=kind, **details)) + '\n')
