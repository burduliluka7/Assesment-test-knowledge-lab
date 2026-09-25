import hashlib
from pathlib import Path
import numpy as np


class Encoder:
    def __init__(self, name, revision, cache, batch=64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, revision=revision, device="cpu")
        self.name = name
        self.revision = revision
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.batch = batch

    def encode(self, texts):
        keys = [
            hashlib.sha256((self.name + self.revision + t).encode()).hexdigest()
            for t in texts
        ]
        missing = {
            k: t
            for k, t in zip(keys, texts)
            if not (self.cache / (k + ".npy")).exists()
        }
        if missing:
            xx = self.model.encode(
                list(missing.values()),
                batch_size=self.batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for k, x in zip(missing, xx):
                np.save(self.cache / (k + ".npy"), x)
        return np.array([np.load(self.cache / (k + ".npy")) for k in keys])


class NLI:
    def __init__(self, cfg):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(
            cfg["nli_model"], revision=cfg["nli_revision"]
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["nli_model"], revision=cfg["nli_revision"], use_safetensors=True
        ).eval()
        self.max_tokens = cfg["label_nli_max_tokens"]
        self.mapping = {
            int(k): v.lower() for k, v in self.model.config.id2label.items()
        }
        assert set(self.mapping.values()) == {"entailment", "neutral", "contradiction"}

    def score(self, pairs):
        result = []
        for start in range(0, len(pairs), 16):
            pp = pairs[start : start + 16]
            encoded = self.tok(
                [a for a, b in pp],
                [b for a, b in pp],
                padding=True,
                truncation="only_first",
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            with self.torch.inference_mode():
                probs = self.model(**encoded).logits.softmax(-1).numpy()
            for (a, b), p in zip(pp, probs):
                n = len(self.tok(a, b, truncation=False)["input_ids"])
                result.append(
                    dict(
                        probabilities={
                            self.mapping[i]: float(x) for i, x in enumerate(p)
                        },
                        label=self.mapping[int(p.argmax())],
                        input_tokens=n,
                        truncated=n > self.max_tokens,
                    )
                )
        return result
