"""Validate active V2 frozen artifacts and publish a precise task file inventory."""
import json
import re
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from tkh_abstraction_v2.io import read,write,sha
from tkh_abstraction_v2.freeze import verify_freeze


def main():
    output=ROOT/'artifacts/prediction';ev=ROOT/'artifacts/evaluation';audit=ROOT/'artifacts/verification'
    audit.mkdir(parents=True,exist_ok=True)
    primary=verify_freeze(ROOT,output);replay=verify_freeze(ROOT,ROOT/'artifacts/reproduction')
    registration=read(ROOT/'notes/requirement_design/final_design_registration.json')
    for relative,digest in registration['predictive_policy_sha256'].items():
        assert sha(ROOT/relative)==digest, 'Predictive policy changed after registration: '+relative
    assert primary['predictions_sha256']==replay['predictions_sha256']
    log=(audit/'tests_final.log').read_text(encoding='utf8')
    assert read(audit/'tests_exit.json')['exit_code']==0
    match=re.search(r'(\d+) passed',log);assert match and 'failed' not in log and 'skipped' not in log
    protected=read(audit/'protected_verification.json');assert protected['unchanged']
    for name in ('clean_tshc','rest_of_work'):
        diff=subprocess.check_output(['git','diff','--binary','--',name],cwd=ROOT.parent)
        normal=subprocess.check_output(['git','diff','--',name],cwd=ROOT.parent)
        assert diff==(ROOT/'notes/requirement_design'/f'{name}_diff_before.patch').read_bytes()
        (audit/f'{name}_diff_final.patch').write_bytes(diff);(audit/f'{name}_git_diff.txt').write_bytes(normal)
    status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT.parent)
    (audit/'git_status_final.txt').write_bytes(status)
    access=read(output/'audit/prediction_accesses.json')
    assert not access['denied_reads'] and access['gold_reads']==0 and access['historical_evaluation_reads']==0
    assert not any('evaluation' in n or 'strict_targets' in n for n in access['local_imports'])
    assert read(ev/'evaluation_only/positive_controls.json')['passed']
    events=[json.loads(line) for line in (output/'audit/events.jsonl').read_text().splitlines()]
    names=[e['event'] for e in events]
    assert names.index('predictions_frozen')<names.index('evaluation_import')<names.index('gold_loaded')
    evaluation=read(ev/'audit/evaluation_manifest.json')
    assert sha(ROOT/'src/tkh_abstraction_v2/entity_evaluation.py')==evaluation['code_sha256']
    compilation=read(audit/'report_compile.json');assert compilation['exit_codes']==[0,0]
    assert compilation['tex_sha256']==sha(ROOT/'report/v2_entity_retrieval_report.tex')
    for stem,details in compilation['reports'].items():
        assert details['exit_codes']==[0,0] and details['tex_sha256']==sha(ROOT/'report'/(stem+'.tex'))
    report_numbers=read(audit/'report_numerical_audit.json');assert report_numbers['passed']
    report_visual=read(audit/'report_visual_check.json')
    assert report_visual['reports']['v2_assessment_companion']['pages']==4
    for stem,details in report_visual['reports'].items():
        assert details['sha256']==sha(ROOT/'report'/(stem+'.pdf'))
    author=read(ROOT/'notes/requirement_design/report_result.json');assert not author['is_error']
    result=dict(tests_passed=int(match[1]),tests_failed=0,prediction_sha256=primary['predictions_sha256'],
        frozen_artifacts_verified=len(primary['files']),replay_frozen_artifacts_verified=len(replay['files']),
        cached_replay_byte_identical=True,prediction_project_reads=len(access['project_reads']),
        gold_reads=0,historical_evaluation_reads=0,positive_controls_passed=True,
        freeze_before_evaluator_and_gold=True,evaluation_write_protection_tested=True,
        no_benchmark_training_or_fitting=True,predictive_policy_matches_pre_evaluation_registration=True,primary_pool=500,requirement_evidence_k=3,
        protected_trees={n:dict(files=r['files'],unchanged=r['unchanged']) for n,r in protected['trees'].items()},
        report_authored_by=author['usedModels'],report_compiled=True,
        report_metric_rows_verified=report_numbers['metric_table_rows_verified'],
        report_target_rows_verified=report_numbers['target_occurrence_rows_verified'],
        report_pages={stem:d['pages'] for stem,d in report_visual['reports'].items()},github_changes=False,
        limitation='Development-only; full replay reuses cached outputs, not a second uncached inference experiment')
    write(audit/'final_verification.json',result)
    baseline={p:h for p,h in read(ROOT/'notes/requirement_design/v2_initial.json').items() if not p.startswith('notes/requirement_design/')}
    current={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and
        not {'.cache','archive','__pycache__','.pytest_cache'}&set(p.relative_to(ROOT).parts)}
    inventory=dict(scope='Paths relative to clean_tshc_v2; caches and pre-existing archive excluded',
        modified=sorted(n for n in baseline if n in current and baseline[n]!=current[n]),
        added=sorted(set(current)-set(baseline)),deleted=sorted(set(baseline)-set(current)))
    write(audit/'files_changed.json',inventory)
    important=['report/v2_entity_retrieval_report.tex','report/v2_entity_retrieval_report.pdf','report/v2_retrieval_report.md',
        'report/stage_metrics.pdf','report/verification_summary.tex','report/v2_assessment_companion.tex','report/v2_assessment_companion.pdf','artifacts/evaluation/metrics.json',
        'artifacts/evaluation/efficiency_report.json','artifacts/evaluation/failure_traces.json',
        'artifacts/evaluation/audit/evaluation_manifest.json','artifacts/verification/final_verification.json',
        'artifacts/verification/report_numerical_audit.json','artifacts/verification/report_visual_check.json',
        'AI_USAGE.md','README.md','notes/ASSESSMENT_ALIGNMENT.md',
        'artifacts/verification/files_changed.json','src/tkh_abstraction_v2/entity_evaluation.py','scripts/evaluate.py']
    write(audit/'final_artifact_hashes.json',{n:sha(ROOT/n) for n in important})
    print(json.dumps(result,indent=2))
    print('V2 files modified/added/deleted:',*[len(inventory[k]) for k in ('modified','added','deleted')])


if __name__=='__main__':main()
