"""Check report tables against the post-freeze evaluation, without changing ranks."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
metrics = json.loads((ROOT / 'artifacts/evaluation/metrics.json').read_text(encoding='utf8'))
handoff = json.loads((ROOT / 'notes/requirement_design/report_handoff.json').read_text(encoding='utf8'))
tex = (ROOT / 'report/v2_entity_retrieval_report.tex').read_text(encoding='utf8')


def clean(value):
    return re.sub(r'\\textbf\{([^}]+)\}', r'\1', value).replace(r'\_', '_').replace(r'$\to$', '\u2192').replace(r'\\', '').strip()


checked = 0
for label in ('tab:candrecall', 'tab:final'):
    section = tex.split(r'\label{' + label + '}', 1)[1].split(r'\end{table}', 1)[0]
    for line in section.splitlines():
        if '&' not in line:
            continue
        cells = [clean(c) for c in line.split('&')]
        arm = cells[0].replace(' (primary)', '')
        if arm.startswith('A6, A7'):
            arm = 'A6'
        if arm not in metrics['systems']:
            continue
        row = metrics['systems'][arm]
        if label == 'tab:candrecall':
            expected = [row['candidate_hits'][str(k)] for k in (50, 100, 200, 500, 1000)] + [row['candidate_union_hits'], row['candidate_union_recall'], row['average_candidate_set_size']]
            tolerances = [0] * 6 + [.00051, .051]
            assert cells[-1] == '--'.join(map(str, row['candidate_size_range'])), arm
        else:
            expected = [row['direct_hits'][str(k)] for k in (1, 5, 10, 25, 50, 100)] + [row['recall'][str(k)] for k in (10, 50, 100)] + [row['MRR'], row['median_gold_rank'], row['mean_gold_rank'], row['missing_gold_ranks'], row['mean_gold_rank_missing_penalized']]
            tolerances = [0] * 6 + [.00051] * 4 + [.051, .051, 0, .051]
        for actual, value, tolerance in zip(cells[1:], expected, tolerances):
            assert abs(float(actual) - value) <= tolerance, (label, arm, actual, value)
        assert len(cells) == len(expected) + (2 if label == 'tab:candrecall' else 1)
        checked += 1

targets = {(qid, t['target']): t for qid, q in handoff['per_question'].items() for t in q['targets']}
count = 0
for line in tex.splitlines():
    if not re.match(r'Q\d+ &', line):
        continue
    cells = [clean(c) for c in line.split('&')]
    row = targets[(cells[0], cells[1])]
    assert cells[2] == row['status']
    for actual, arm in zip(cells[3:8], ('A5', 'A6', 'A7', 'A7_NoNLI', 'A7_SingleEvidence')):
        assert actual == str(row['ranks'][arm] or '--'), (cells[:2], arm, actual)
    count += 1
assert count == 63
result = dict(metric_table_rows_verified=checked, target_occurrence_rows_verified=count, passed=True,
              scope='Numerical tables against frozen evaluation handoff; narrative checked separately')
(ROOT / 'artifacts/verification/report_numerical_audit.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(result))
