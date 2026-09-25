"""Generic candidate roles, explicitly separate from immutable graph/name discovery."""
import re
from .target_mapping import normalize,contains_name
from .diffusion import safe_mention_name

COMPARISON=re.compile(r'\b(outperforms?|outperformed by|better than|worse than|improves? over|fails? compared with|compared (?:with|to))\b',re.I)

def classify_attribution(item,names):
    text=normalize(item['text']); names=[n for n in names if safe_mention_name(n) and contains_name(item['text'],n)]
    role='UNCERTAIN'; reason='No safe subject assertion inferred from graph co-membership'
    comparison=None
    if names:
        for m in COMPARISON.finditer(text):
            left=text[:m.start()]; right=text[m.end():]
            on_left=any(contains_name(left,n) for n in names); on_right=any(contains_name(right,n) for n in names)
            comparison=dict(predicate=m.group(),candidate_before=on_left,candidate_after=on_right)
            if on_left==on_right or re.search(r'\b(?:not|never|no)\b',left[-40:]) or m.group().startswith('compared'):
                role='UNCERTAIN'; reason='Comparison role or polarity is ambiguous'
            else:
                reverse=m.group() in {'worse than','outperformed by'} or m.group().startswith('fail')
                winner=on_right if reverse else on_left
                role='COMPARISON_WINNER' if winner else 'COMPARISON_BASELINE'; reason='Candidate side of generic comparative predicate'
            break
        if len(list(COMPARISON.finditer(text)))>1:
            role='UNCERTAIN'; reason='Multiple comparative predicates; single-clause direction is insufficient'
        if comparison is None:
            listed=bool(re.search(r'\b(?:including|such as|e g|among|include|includes)\b',text))
            subject=any(re.match(r'^(?:the\s+)?'+re.escape(normalize(n))+r'(?![\w+\-])',text) for n in names)
            if listed and not subject: role='MENTION_ONLY'; reason='Candidate appears in an illustrative list'
            elif subject and item['node_type']=='claim': role='DIRECT_SUBJECT_SUPPORT'; reason='Candidate begins the claim; polarity assessed by NLI separately'
            elif item.get('kind')=='safe_explicit_name' or item['node_type']=='claim': role='EXPLICIT_NAME_SUPPORT'; reason='Safe name mention, no comparison/list marker; subject remains heuristic'
            else: role='MENTION_ONLY'; reason='Named title/label is not an asserted task capability'
            if re.search(r'\b(?:not|never|cannot|fails?)\b',text):
                role='UNCERTAIN'; reason='Explicit negation/failure requires condition-level interpretation; not unconditional positive support'
    elif item.get('kind')=='short_path_source':
        role='INDIRECT_GRAPH_NEIGHBOR'; reason='Path source does not name candidate: diagnostic only'
    elif item.get('kind')=='provenance_title_fallback':
        role='INDIRECT_GRAPH_NEIGHBOR'; reason='Shared provenance title without candidate identity'
    return dict(category=role,reason=reason,matched_names=names,comparison=comparison,
        graph_support_kind=item.get('support_kind',item['kind']),graph_relation=item['kind'],
        co_membership_only=item.get('support_kind')=='comembership_only',path_only=item.get('kind')=='short_path_source' and not names,
        negation_detected=bool(re.search(r'\b(?:not|never|without|cannot|fails?)\b',text)),
        numerical=bool(re.search(r'\d',text)),conjunction=bool(re.search(r'\b(?:and|or)\b',text)))
