"""Additive grammar corrections for CR6; historical decomposition stays frozen."""
from dataclasses import dataclass, field
from copy import deepcopy
import re
from .question_decomposition import QuestionDecomposition, decompose_question
from .evidence_attribution import classify_attribution
from .target_mapping import contains_name


@dataclass
class ConditionedQuestionDecomposition(QuestionDecomposition):
    answer_qualifiers: list = field(default_factory=list)


def decompose_conditioned_question(question, declared_type='A'):
    old = decompose_question(question, declared_type)
    d = ConditionedQuestionDecomposition(**deepcopy(old.to_dict()))
    # Recover only modifiers before a recognized answer noun. Dates keep offsets.
    head = d.source_spans.get('answer_type')
    qualifiers = []
    if head and d.answer_type:
        end = head['start']; mask = list(question[:end])
        for t in d.source_spans['temporal_condition']:
            for i in range(t['start'], min(t['end'], end)): mask[i] = ' '
        visible = ''.join(mask)
        # A dangling coordination before the noun is not a modifier phrase.
        if re.search(r'\b(?:and|or)\s*$',visible,re.I):
            visible=''
            d.warnings.append('Coordinated answer head is ambiguous; no qualifiers extracted')
        framing = re.match(r'^\s*(?:what|which)\s+(?:(?:are|is)\s+)?(?:(?:the|a|an|leading|best|available|current|most|suitable)[\s(),]+)*[\s(),]*', visible, re.I)
        if framing:
            start = framing.end()
            # Split hyphenated adjectival modifiers; keep ordinary compound nouns intact.
            for m in re.finditer(r'[\w]+(?:[-\u2011\u2010][\w]+)+|[^(),]+', visible[start:end]):
                a, b = start+m.start(), start+m.end()
                while a < b and question[a].isspace(): a += 1
                while b > a and question[b-1].isspace(): b -= 1
                if a < b and re.search(r'\w', question[a:b]):
                    qualifiers.append(dict(text=question[a:b], source_span=question[a:b], start=a, end=b,
                                           kind='answer_qualifier', connector=''))
    # Preserve an entire coordinated clause if its final part carries a shared modal.
    merged = []; i = 0
    while i < len(d.components):
        first = d.components[i]; j = i
        finite=re.compile(r'\b(?:must|should|shall|can|will|may|is|are|was|were|has|have|does|do|supports?|requires?|needs?|allows?|enables?|provides?)\b',re.I)
        if first['kind'] == 'requirement' and not finite.search(first['text']):
            while j+1 < len(d.components):
                nxt = d.components[j+1]
                gap = question[d.components[j]['end']:nxt['start']]
                if nxt['kind'] != 'requirement' or not re.fullmatch(r'\s*,?\s*and\s+', gap, re.I): break
                j += 1
                if re.search(r'\b(?:must|should|shall|can|will|may)\s+(?:be|remain|stay)\b', nxt['text'], re.I):
                    first = dict(first, text=question[first['start']:nxt['end']],
                                 source_span=question[first['start']:nxt['end']], end=nxt['end'])
                    break
                if finite.search(nxt['text']): break
            else: j = i
            if first['end'] == d.components[i]['end']: j = i
        merged.append(first); i = j+1
    d.components = merged + qualifiers
    for i, c in enumerate(d.components): c['id'] = f'component_{i}'
    d.requirements = [c['text'] for c in merged if c['kind'] == 'requirement']
    d.source_spans['requirements'] = [{k:c[k] for k in ('text','source_span','start','end')} for c in merged if c['kind']=='requirement']
    d.answer_qualifiers = [c['text'] for c in qualifiers]
    d.source_spans['answer_qualifiers'] = [{k:c[k] for k in ('text','source_span','start','end')} for c in qualifiers]
    return d


def classify_conditioned_attribution(item, names):
    result = classify_attribution(item, names)
    # Sentence-initial participial subject + finite predicate is not an example list.
    # Keep comparison, negation, other list markers, and ambiguous uses untouched.
    text = item['text']
    predicate=re.search(r'\b(?:improves|enhances|increases|reduces|enables|prevents|changes)\b',text,re.I)
    if (result['category'] == 'MENTION_ONLY' and item['node_type'] == 'claim'
            and re.match(r'^\s*Including\b', text, re.I)
            and predicate
            and not any(contains_name(text[:predicate.start()],n) for n in result['matched_names'])
            and any(contains_name(text[predicate.end():],n) for n in result['matched_names'])
            and ',' not in text[:predicate.start()]
            and len(re.findall(r'\bincluding\b', text, re.I)) == 1
            and not re.search(r"\b(?:such as|among|include|includes|not|no|never|without|cannot|lacks?|fails?|than|versus|relative to|unlike|over|compared)\b|n['\u2019]t\b", text, re.I)):
        result = dict(result, category='EXPLICIT_NAME_SUPPORT',
                      reason='Sentence-initial participial subject with finite predicate; safe name support, not direct candidate subject')
    return result
