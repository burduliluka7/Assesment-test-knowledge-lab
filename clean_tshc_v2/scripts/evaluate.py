"""Verify the completed prediction freeze BEFORE importing any evaluation module."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from tkh_abstraction_v2.io import read, sha, event


from tkh_abstraction_v2.freeze import verify_freeze, protect_evaluation


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/prediction')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Evaluation output must stay inside V2')
    freeze = protect_evaluation(ROOT, output)
    event(output / 'audit/events.jsonl', 'evaluation_import', predictions_sha256=freeze['predictions_sha256'])
    from tkh_abstraction_v2.entity_evaluation import evaluate
    evaluate(ROOT, [args.output])
    verify_freeze(ROOT, output)


if __name__ == '__main__':
    main()
