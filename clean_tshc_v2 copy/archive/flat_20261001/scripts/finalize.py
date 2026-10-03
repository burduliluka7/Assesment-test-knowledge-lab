"""Verify and collect final artifacts without recomputing or changing rankings."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from tkh_abstraction_v2.io import read, write, sha
from evaluate import verify_freeze


def main():
    out = ROOT / 'outputs'
    freeze = verify_freeze(ROOT)
    protected = read(out / 'audit/protected_verification.json')
    assert protected['unchanged'], 'Protected files changed'
    assert (out / 'retrieval/predictions.sha256').read_text().strip() == freeze['predictions_sha256']
    tests_raw = (out / 'audit/tests_final.log').read_bytes()
    tests = tests_raw.decode('utf-16' if tests_raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf8')
    assert '19 passed' in tests and 'failed' not in tests
    models = read(out / 'audit/model_availability.json')
    for model in models.values():
        if model['status'] == 'available':
            for relative, digest in model['files'].items():
                assert sha(ROOT / model['path'] / relative) == digest, relative
    inputs = {}
    for name in ('tkh_collection10.json', 'questions.csv'):
        inputs[name] = sha(ROOT / 'data' / name) == sha(ROOT.parent / 'clean_tshc/data' / name)
    assert all(inputs.values())
    events = [json.loads(line) for line in (out / 'audit/events.jsonl').read_text().splitlines()]
    kinds = [e['event'] for e in events]
    assert kinds.index('predictions_frozen') < kinds.index('evaluation_import') < kinds.index('gold_loaded')
    result = dict(tests='19 passed', protected_reference_unchanged=True, protected_files=protected['files'],
        prediction_sha256=freeze['predictions_sha256'], frozen_artifacts_verified=len(freeze['files']),
        model_file_hashes_verified=True, copied_inputs_match_reference=inputs,
        prediction_before_evaluation_before_gold=True, git_history_operations_performed=False,
        limitation='Fixed-input deterministic inference tested; no full duplicate benchmark run claimed')
    write(out / 'audit/final_verification.json', result)
    report = ROOT / 'report/v2_retrieval_report.md'
    content = report.read_text(encoding='utf8').split('\n## Final verification\n')[0]
    report.write_text(content + '\n## Final verification\n\n'
        '19 tests passed. All prediction/source/input freeze hashes and cached model-file hashes verify. '
        'Copied graph and question files match the reference. Prediction freeze precedes evaluator import and gold loading. '
        '**CLEAN_TSHC UNCHANGED: YES** — all 1,056 file hashes and the complete before/after manifest bytes match. '
        'No git staging, commits, pushes, resets, or history changes were performed. '
        'The exhaustive file list, including generated artifacts and caches, is [new_files.txt](../outputs/audit/new_files.txt).\n', encoding='utf8')
    inventory = out / 'audit/new_files.txt'
    paths = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    paths.add(inventory.relative_to(ROOT).as_posix())
    inventory.write_text('\n'.join('clean_tshc_v2/' + p for p in sorted(paths)) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2))
    print('New-file inventory:', len(paths), 'files')


if __name__ == '__main__':
    main()
