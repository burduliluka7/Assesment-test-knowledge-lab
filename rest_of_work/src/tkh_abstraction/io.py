import csv
import hashlib
import json
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        sample=f.read(4096); f.seek(0)
        return list(csv.DictReader(f,delimiter=';' if ';' in sample.splitlines()[0] else ','))

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def locate(directory, name):
    matches=sorted(Path(directory).rglob(name)) if Path(directory).exists() else []
    if not matches:
        raise FileNotFoundError(f"Missing supplied assessment file {name}. Place the supplied files in {directory}; no replacement dataset is downloaded.")
    if len(matches)>1:
        raise ValueError(f"Ambiguous {name}: {matches}; configure a specific data_dir")
    return matches[0]
