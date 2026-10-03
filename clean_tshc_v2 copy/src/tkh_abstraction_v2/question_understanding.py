"""Question-only intent extraction; no answer lookup or scientific fact generation."""
from abc import ABC, abstractmethod
import json
from .extractive import decompose_question

PARSER_PROMPT = '''You are a scientific query parser. Do not answer the question.
Return STRICT JSON only with exactly these keys: answer_type, core_task,
requirements, exclusions, temporal_condition, search_query, warnings.
Extract answer entity type, main task, positive requirements, exclusions, time
constraint, and a concise retrieval query. Do not introduce scientific facts or
entity names absent from the question. Use contiguous question substrings for
core_task, requirements, exclusions, and temporal_condition. requirements,
exclusions and warnings are arrays of strings. Other fields are strings or null.'''


class QuestionParser(ABC):
    @abstractmethod
    def parse(self, question):
        raise NotImplementedError


def representations(d):
    d['structured_query'] = '\n'.join([
        'answer type: ' + (d['answer_type'] or 'scientific entity'),
        'task: ' + (d['core_task'] or d['original_question']),
        *['requirement: ' + r for r in d['requirements']],
        *['exclude: ' + r for r in d['exclusions']],
        *(['time: ' + d['temporal_condition']] if d['temporal_condition'] else []),
    ])
    return d


class ExtractiveParser(QuestionParser):
    def parse(self, question):
        d = decompose_question(question).to_dict()
        d['parser'] = 'deterministic_extractive'
        d['search_query'] = '; '.join(x for x in [d['answer_type'], d['core_task'], *d['requirements'], *['without ' + s for s in d['exclusions']]] if x)
        if not d['answer_type']:
            d['warnings'].append('Unrecognized answer family; use all non-author entities')
        if d['temporal_condition']:
            d['warnings'].append('Temporal phrase recorded; primary snapshot remains publication-visible year 2026, no month precision')
        return representations(d)


class StrictJSONParser(QuestionParser):
    """Optional injected LOCAL instruction-model callable(prompt)->JSON string.

    No model or endpoint is silently selected. Malformed/nonextractive output
    fails explicitly; the primary experiment uses ExtractiveParser.
    """
    def __init__(self, generate):
        self.generate = generate

    def parse(self, question):
        d = json.loads(self.generate(PARSER_PROMPT + '\nQuestion: ' + question))
        keys = {'answer_type', 'core_task', 'requirements', 'exclusions', 'temporal_condition', 'search_query', 'warnings'}
        if set(d) != keys:
            raise ValueError('Unexpected parser JSON schema')
        for key in ('requirements', 'exclusions', 'warnings'):
            if not isinstance(d[key], list) or not all(isinstance(s, str) for s in d[key]):
                raise ValueError('Expected list of strings: ' + key)
        for key in keys - {'requirements', 'exclusions', 'warnings'}:
            if d[key] is not None and not isinstance(d[key], str):
                raise ValueError('Expected string or null: ' + key)
        for s in [d['core_task'], d['temporal_condition'], *d['requirements'], *d['exclusions']]:
            if s and s not in question:
                raise ValueError('Parser introduced non-extractive content')
        import re
        allowed = set(re.findall(r'\w+', question.lower())) | {'method', 'dataset', 'article', 'metric', 'technique', 'component', 'task', 'problem', 'for', 'without', 'and'}
        if set(re.findall(r'\w+', (d['search_query'] or '').lower())) - allowed:
            raise ValueError('Rewrite introduced unsupported vocabulary')
        d.update(original_question=question, parser='local_instruction_strict_json')
        return representations(d)
