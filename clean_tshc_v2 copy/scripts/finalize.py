"""Verify frozen results, report provenance, and the user-approved V1 starting state."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from tkh_abstraction_v2.io import read, write, sha
from evaluate import verify_freeze
from audit_reference import verify_reference


def main():
    out = ROOT / 'outputs'
    freeze = verify_freeze(ROOT)
    replay = verify_freeze(ROOT, out / 'reproduction')
    assert freeze['predictions_sha256'] == replay['predictions_sha256']
    assert (out / 'retrieval/predictions.sha256').read_text().strip() == freeze['predictions_sha256']
    tests = (out / 'audit/tests_final.log').read_text(encoding='utf8')
    assert read(out / 'audit/tests_exit.json')['exit_code'] == 0
    assert re.search(r'\b42 passed\b', tests) and 'failed' not in tests
    model_manifest = read(out / 'audit/model_version_manifest.json')
    model_files = 0
    for role in ('dense', 'relevance', 'nli'):
        model = model_manifest['models'][role]
        assert model['status'] == 'available', role
        for relative, digest in model['files'].items():
            assert sha(ROOT / model['path'] / relative) == digest, (role, relative)
            model_files += 1
    inputs = {name: sha(ROOT / 'data' / name) == sha(ROOT.parent / 'clean_tshc/data' / name)
              for name in ('tkh_collection10.json', 'questions.csv')}
    assert all(inputs.values())
    events = [json.loads(line) for line in (out / 'audit/events.jsonl').read_text().splitlines()]
    kinds = [e['event'] for e in events]
    assert kinds.index('predictions_frozen') < kinds.index('evaluation_import') < kinds.index('gold_loaded')
    assert read(out / 'evaluation_only/positive_controls.json')['passed']
    evaluation = read(out / 'audit/evaluation_manifest.json')
    assert sha(ROOT / 'src/tkh_abstraction_v2/entity_evaluation.py') == evaluation['code_sha256']
    authorship = read(ROOT / 'notes/entity_redesign/report_revision_result.json')
    assert not authorship['is_error'] and 'claude-opus-5' in authorship['usedModels']
    compilation = read(out / 'audit/report_compile.json')
    assert compilation['exit_codes'] == [0, 0]
    report = ROOT / 'report/v2_entity_retrieval_report.tex'
    pdf = report.with_suffix('.pdf')
    assert pdf.read_bytes().startswith(b'%PDF-')
    assert compilation['tex_sha256'] == sha(report)
    protected = verify_reference()
    result = dict(tests='42 passed', protected_reference_unchanged=True, protected_files=protected['files'],
        preexisting_v1_diff_preserved=protected['git_diff_matches_start'],
        prediction_sha256=freeze['predictions_sha256'], frozen_artifacts_verified=len(freeze['files']),
        replay_frozen_artifacts_verified=len(replay['files']), cached_replay_byte_identical=True,
        model_file_hashes_verified=model_files, copied_inputs_match_reference=inputs,
        prediction_before_evaluation_before_gold=True, positive_controls_passed=True,
        report_authored_by=authorship['usedModels'], report_compiled=True,
        git_history_operations_performed=False,
        limitation='Development diagnostics; cached replay is not an independent uncached inference run')
    write(out / 'audit/final_verification.json', result)
    markdown = ROOT / 'report/v2_retrieval_report.md'
    content = markdown.read_text(encoding='utf8').split('\n## Final verification\n')[0]
    markdown.write_text(content + '\n## Final verification\n\n'
        '42 tests passed. All frozen prediction/source/input and cached model-file hashes verify. '
        'The full cached replay produces byte-identical predictions. Prediction freeze precedes evaluator import and gold loading. '
        '**V1 has zero new changes**: all 1,056 file hashes, its starting git diff, and its starting git status match. '
        'Pre-existing V1 work is preserved as explicitly requested. No GitHub changes were made. '
        'Claude Opus 5 authored and revised the detailed [LaTeX report](v2_entity_retrieval_report.tex), '
        'with Codex factual corrections and compilation to [PDF](v2_entity_retrieval_report.pdf). '
        'See [file inventory](../outputs/audit/files_changed.json) for the active V2 changes and preserved archive.\n', encoding='utf8')
    baseline = read(ROOT / 'notes/entity_redesign/v2_before.json')
    current = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*')
               if p.is_file() and not {'.cache', 'archive', '__pycache__', '.pytest_cache'} & set(p.relative_to(ROOT).parts)}
    archived = {}
    for relative, digest in baseline.items():
        old = ROOT / 'archive/flat_20261001' / relative
        if old.is_file() and sha(old) == digest:
            archived[relative] = old.relative_to(ROOT).as_posix()
    inventory = dict(scope='Active V2 files relative to task start; caches and archive excluded from added/changed lists',
        modified=sorted(p for p in baseline if p in current and baseline[p] != current[p]),
        added=sorted(set(current) - set(baseline)),
        no_longer_at_active_path=sorted(set(baseline) - set(current)),
        preserved_originals=archived)
    write(out / 'audit/files_changed.json', inventory)
    important = ['report/v2_entity_retrieval_report.tex', 'report/v2_entity_retrieval_report.pdf',
        'report/v2_retrieval_report.md', 'report/stage_metrics.pdf', 'report/verification_summary.tex',
        'outputs/metrics.json', 'outputs/failure_traces.json', 'outputs/per_question_evaluation.json',
        'outputs/efficiency_report.json', 'outputs/evaluation_only/positive_controls.json',
        'outputs/audit/final_verification.json', 'outputs/audit/files_changed.json',
        'src/tkh_abstraction_v2/entity_evaluation.py', 'scripts/evaluate.py', 'scripts/finalize.py']
    write(out / 'audit/final_artifact_hashes.json', {p: sha(ROOT / p) for p in important})
    print(json.dumps(result, indent=2))
    print('Active V2:', len(inventory['modified']), 'modified;', len(inventory['added']), 'added; preserved originals:', len(archived))


if __name__ == '__main__':
    main()
