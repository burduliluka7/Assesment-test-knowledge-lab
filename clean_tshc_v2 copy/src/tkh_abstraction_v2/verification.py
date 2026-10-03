"""Top-set requirement verification; NLI is explicitly a proxy, never relevance."""
import time

STATUSES = {'SUPPORTED', 'NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE', 'CONTRADICTED', 'NOT_EVALUABLE'}


def status_from_probabilities(probabilities, evidence_present, threshold=.7, exclusion=False):
    if not evidence_present:
        return 'NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE'
    if probabilities is None:
        return 'NOT_EVALUABLE'
    ent, con = probabilities['entailment'], probabilities['contradiction']
    if exclusion:
        ent, con = con, ent
    if ent >= threshold and con < 1-threshold:
        return 'SUPPORTED'
    if con >= threshold and ent < 1-threshold:
        return 'CONTRADICTED'
    return 'NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE'


class NliVerifier:
    def __init__(self, path):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True, use_safetensors=True).eval()
        self.labels = {int(i): v.lower() for i, v in self.model.config.id2label.items()}
        if set(self.labels.values()) != {'entailment', 'neutral', 'contradiction'}:
            raise ValueError('NLI verifier requires explicit entailment/neutral/contradiction labels')

    def score(self, pairs):
        import torch
        result = []
        for start in range(0, len(pairs), 16):
            subset = pairs[start:start+16]
            tokens = self.tokenizer([a for a,b in subset], [b for a,b in subset], padding=True, truncation='only_first', max_length=512, return_tensors='pt')
            with torch.inference_mode():
                values = self.model(**tokens).logits.softmax(-1).numpy()
            for pair, row, mask in zip(subset, values, tokens['attention_mask']):
                result.append(dict(probabilities={self.labels[i]: float(v) for i,v in enumerate(row)},
                    input_tokens=int(mask.sum()), full_tokens=len(self.tokenizer(*pair)['input_ids']), proxy_only=True))
        return result


def verify_requirements(intent, entity_ids, index, retriever, nli, cfg):
    components = [dict(kind='core_task', text=intent['core_task'] or intent['original_question']),
        *[dict(kind='requirement', text=s) for s in intent['requirements']],
        *[dict(kind='exclusion', text=s) for s in intent['exclusions']]]
    queries = retriever.encoder.encode([c['text'] for c in components])
    matrix = retriever.evidence_vectors @ queries.T
    loc = {i: j for j, i in enumerate(retriever.evidence_ids)}
    result, pending, pairs = {}, [], []
    for eid in entity_ids:
        entity = index['entities'][eid]
        candidates = {r['evidence_id']: r for rows in entity['evidence'].values() for r in rows
                      if r['candidate_specific'] and r['field'] not in ('articles', 'graph_evidence')}
        records = []
        for j, component in enumerate(components):
            selected = max(candidates, key=lambda i: (float(matrix[loc[i], j]), i), default=None)
            text = index['items'][selected]['text'] if selected else None
            hypothesis = entity['name'] + (' involves ' if component['kind'] == 'exclusion' else ' supports ') + component['text'] + '.'
            record = dict(component, evidence_id=selected, evidence_text=text,
                evidence_link=candidates.get(selected), similarity=float(matrix[loc[selected], j]) if selected else None,
                hypothesis=hypothesis, premise=None, probabilities=None,
                status=status_from_probabilities(None, selected is not None), interpretation='NLI proxy over available evidence, not scientific adjudication')
            if selected:
                record['premise'] = 'Candidate: ' + entity['name'] + '. Relation: ' + candidates[selected]['support_kind'] + '. Evidence: ' + text
                if nli:
                    pairs.append((record['premise'], hypothesis))
                    pending.append(record)
            records.append(record)
        result[eid] = dict(components=records, missing_evidence_is_not_contradiction=True)
    start = time.perf_counter()
    if nli and pairs:
        for record, scores in zip(pending, nli.score(pairs)):
            record.update(scores)
            record['status'] = status_from_probabilities(record['probabilities'], True, cfg['nli_threshold'], record['kind']=='exclusion')
    for eid, record in result.items():
        statuses = [c['status'] for c in record['components']]
        record['tier'] = 0 if statuses and all(s == 'SUPPORTED' for s in statuses) else 2 if 'CONTRADICTED' in statuses else 1
    return result, dict(nli_pairs=len(pairs), requirement_components=len(entity_ids)*len(components),
        evidence_comparisons=int(matrix.size), wall_seconds=time.perf_counter()-start, nli_available=nli is not None)


def verification_order(ranking, verification, k=20):
    prefix = ranking[:k]
    head = sorted(enumerate(prefix), key=lambda p: (verification[p[1]['entity_id']]['tier'], p[0]))
    return [dict(row, stage='requirement_verification', verification_tier=verification[row['entity_id']]['tier']) for _,row in head] + ranking[k:]
