from pathlib import Path
import hashlib
import time
import numpy as np


class Encoder:
    def __init__(self, path, cache, batch=32):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(path), device='cpu', local_files_only=True)
        self.model.max_seq_length = 512
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.batch = batch
        self.fingerprint = str(Path(path).name)

    def encode(self, texts):
        if not texts:
            return np.empty((0, 384))
        key = hashlib.sha256((self.fingerprint + '\0' + '\0'.join(texts)).encode()).hexdigest()
        file = self.cache / (key + '.npy')
        if file.exists():
            return np.load(file)
        values = self.model.encode(texts, batch_size=self.batch, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
        np.save(file, values)
        return values


def pack(tokenizer, query, blocks, limit=512):
    identity = blocks[0]
    if identity['field'] != 'identity':
        raise ValueError('Candidate identity must be first')
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    # Preserve identity in full; shorten the question first only if necessary.
    identity_tokens = len(tokenizer.encode(identity['text'], add_special_tokens=False))
    available = limit - identity_tokens - tokenizer.num_special_tokens_to_add(pair=True)
    if available < 1:
        raise ValueError('Candidate identity exceeds model token limit')
    query_truncated = len(query_ids) > available
    if query_truncated:
        query = tokenizer.decode(query_ids[:available], skip_special_tokens=True)
    included, omitted = [identity], []
    document = identity['text']
    for index, block in enumerate(blocks[1:], 1):
        trial = document + '\n' + block['text']
        if len(tokenizer(query, trial, add_special_tokens=True)['input_ids']) > limit:
            omitted = [dict(b, reason='priority_prefix_token_limit') for b in blocks[index:]]
            break
        included.append(block)
        document = trial
    count = len(tokenizer(query, document, add_special_tokens=True)['input_ids'])
    if count > limit:
        raise ValueError('Packing exceeded model limit')
    return query, document, dict(included=included, omitted=omitted, token_count=count, query_truncated=query_truncated)


class RelevanceCrossEncoder:
    def __init__(self, path, limit=512):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True, use_safetensors=True).eval()
        labels = {str(v).lower() for v in self.model.config.id2label.values()}
        if self.model.config.num_labels != 1 or labels & {'entailment', 'neutral', 'contradiction'}:
            raise ValueError('Requires a relevance regression model; NLI is not a relevance cross-encoder')
        self.limit = min(limit, self.model.config.max_position_embeddings)

    def score(self, query, cards, batch=16):
        import torch
        prepared = [pack(self.tokenizer, query, c['blocks'], self.limit) for c in cards]
        result = {}
        for start in range(0, len(cards), batch):
            subset = prepared[start:start + batch]
            now = time.perf_counter()
            encoded = self.tokenizer([s[0] for s in subset], [s[1] for s in subset], padding=True, truncation=False, return_tensors='pt')
            with torch.inference_mode():
                scores = self.model(**encoded).logits[:, 0].numpy()
            seconds = time.perf_counter() - now
            for card, (_, _, detail), value in zip(cards[start:start + batch], subset, scores):
                result[card['node_id']] = dict(detail, relevance_score=float(value), batch_runtime_seconds=seconds,
                    batch_size=len(subset), runtime_seconds_amortized=seconds / len(subset))
        return result
