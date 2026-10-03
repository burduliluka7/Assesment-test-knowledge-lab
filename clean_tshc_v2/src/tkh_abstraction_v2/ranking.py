import math
import re
from collections import Counter
import numpy as np




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






