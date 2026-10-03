"""Supplement the verbatim question; preserve pre-head modifiers and coordinated families."""
import re
from .question_understanding import ExtractiveParser
from .extractive import DATE


def parse_intent(question, snapshot=2026):
    d = ExtractiveParser().parse(question)
    head = d.get('source_spans', {}).get('answer_type')
    modifiers = ''
    answer_types = [d['answer_type']] if d['answer_type'] else []
    if head:
        prefix = DATE.sub('', question[:head['start']])
        modifiers = re.sub(r'\b(?:what|which|are|is|the|a|an|leading|best|most|suitable|available|current)\b', '', prefix, flags=re.I)
        modifiers = ' '.join(modifiers.replace('(', ' ').replace(')', ' ').split())
        first = head['text'].lower().rstrip('s')
        if not answer_types:
            answer_types = [first]
        coordinate = re.match(r'\s+(?:and|or)\s+(datasets?|benchmarks?|methods?|models?|articles?|metrics?|techniques?)\b', question[head['end']:], re.I)
        if coordinate:
            answer_types.append(coordinate[1].lower().rstrip('s'))
    answer_types = sorted(set(answer_types))
    d['answer_types'] = answer_types
    d['preserved_modifiers'] = modifiers
    d['parsed_only_query'] = '; '.join(s for s in [modifiers, d['search_query']] if s)
    d['structured_query'] = '\n'.join(['answer families: ' + ', '.join(answer_types),
        'scientific modifiers: ' + modifiers, d['structured_query']])
    d['combined_query'] = 'Original question: ' + question + '\nStructured intent:\n' + d['structured_query']
    d['temporal_cutoff'] = snapshot
    condition = d.get('temporal_condition')
    if condition:
        years = [int(y) for y in re.findall(r'\b(?:19|20)\d{2}\b', condition)]
        if years and re.search(r'\b(?:by|before|until|as of)\b', condition, re.I):
            year = min(years)
            if re.fullmatch(r'before\s+\d{4}', condition, re.I):
                year -= 1
            d['temporal_cutoff'] = min(snapshot, year)
        d['warnings'].append('Only publication-year visibility can be enforced; dates within a year are not resolved by the source data')
    assert d['original_question'] == question
    return d
