"""Verify the completed prediction freeze BEFORE importing any evaluation module."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from tkh_abstraction_v2.io import read, sha, event


def verify_freeze(root, output=None):
    output = output or root / 'outputs'
    freeze = read(output / 'audit/freeze.json')
    prediction = output / 'retrieval/predictions.json'
    if sha(prediction) != freeze['predictions_sha256']:
        raise ValueError('Prediction hash does not match freeze')
    for relative, digest in freeze['files'].items():
        if sha(root / relative) != digest:
            raise ValueError('Frozen artifact changed: ' + relative)
    events = [__import__('json').loads(line) for line in (output / 'audit/events.jsonl').read_text().splitlines()]
    if not any(e['event'] == 'predictions_frozen' and e['sha256'] == freeze['predictions_sha256'] for e in events):
        raise ValueError('Missing prediction freeze event')
    return freeze


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Evaluation output must stay inside V2')
    freeze = verify_freeze(ROOT, output)
    event(output / 'audit/events.jsonl', 'evaluation_import', predictions_sha256=freeze['predictions_sha256'])
    from tkh_abstraction_v2.entity_evaluation import evaluate
    evaluate(ROOT, [args.output])


if __name__ == '__main__':
    main()
