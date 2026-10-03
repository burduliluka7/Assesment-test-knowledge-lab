import math
import re
from collections import Counter
import numpy as np


def rank(ids, scores):
    return [dict(node_id=i, score=float(s)) for i, s in sorted(zip(ids, scores), key=lambda p: (-float(p[1]), p[0]))]


def tokens(text):
    return re.findall(r'[\w]+(?:[-.][\w]+)*', text.casefold())


class BM25:
    def __init__(self, texts, k1=1.5, b=.75):
        self.docs = [Counter(tokens(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs])
        self.avg = float(np.mean(self.lengths)) or 1.
        self.k1, self.b = k1, b
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (len(texts) - n + .5) / (n + .5)) for t, n in df.items()}

    def score(self, query):
        result = np.zeros(len(self.docs))
        for term in sorted(set(tokens(query))):
            tf = np.array([d[term] for d in self.docs])
            result += self.idf.get(term, 0) * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths / self.avg))
        return result


def rrf(*rankings, k=60):
    values = {}
    for ranking in rankings:
        for r, row in enumerate(ranking, 1):
            values[row['node_id']] = values.get(row['node_id'], 0.) + 1 / (k + r)
    return rank(list(values), list(values.values()))


def replace_prefix(ranking, scores, k):
    prefix = ranking[:k]
    expected = {r['node_id'] for r in prefix}
    if set(scores) != expected:
        raise ValueError('Reranker must score exactly the frozen prefix')
    reranked = rank(list(scores), list(scores.values()))
    return [dict(r, score_stage='reranker') for r in reranked] + [dict(r, score_stage='first_stage_tail') for r in ranking[k:]]


def requirement_score(core, requirements, exclusions, type_prior=1.):
    # Rectified cosine similarities are heuristic support, not probabilities.
    gm = math.exp(sum(math.log(max(v, 1e-6)) for v in requirements) / len(requirements)) if len(requirements) else 1.
    penalty = math.prod(1 - max(0., min(1., float(v))) for v in exclusions)
    return float(core * gm * penalty * type_prior), float(gm), float(penalty)
