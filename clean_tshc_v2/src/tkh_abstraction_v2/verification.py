"""Top-set requirement verification; NLI is explicitly a proxy, never relevance."""

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

