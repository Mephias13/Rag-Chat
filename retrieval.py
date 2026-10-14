import hashlib
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

import llm
from kb import tokens

CACHE = Path("data/cache/embeddings.npz")
BATCH = 32
METHODS = ("bm25_raw", "bm25_stem", "dense", "hybrid")


_store = None


def embed_texts(texts):
    """L2-normalized embeddings, cached on disk by text hash, so cosine is a dot product."""
    global _store
    if _store is None:
        _store = {}
        if CACHE.exists():
            with np.load(CACHE) as f:
                _store = dict(zip(f["keys"].tolist(), f["vecs"]))
    store = _store
    keys = [hashlib.sha1(f"{llm.EMBED_MODEL}\n{t}".encode()).hexdigest() for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in store]
    for s in range(0, len(missing), BATCH):
        idx = missing[s:s + BATCH]
        vecs = np.array(llm.embed([texts[i] for i in idx]), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store.update(zip((keys[i] for i in idx), vecs))
    if missing:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, keys=np.array(list(store)), vecs=np.stack(list(store.values())))
    return np.stack([store[k] for k in keys])


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for pos, article in enumerate(ranking, 1):
            scores[article] = scores.get(article, 0.0) + 1.0 / (k + pos)
    return sorted(scores, key=scores.get, reverse=True)


def rank_articles(articles, scores):
    """Chunk scores -> article ids, best first; an article is as good as its best chunk."""
    seen, ranking = set(), []
    for i in np.argsort(-scores, kind="stable"):
        if articles[i] not in seen:
            seen.add(articles[i])
            ranking.append(articles[i])
    return ranking


class Retriever:
    def __init__(self, chunks, embed_fn=embed_texts):
        self.chunks = chunks
        self.articles = [c["article"] for c in chunks]
        self.embed_fn = embed_fn
        self.bm25 = {
            stem: BM25Okapi([tokens(c["text"], stem=stem) for c in chunks]) for stem in (False, True)
        }
        self.vecs = embed_fn([c["text"] for c in chunks])

    def rank(self, query, method):
        if method == "bm25_raw":
            scores = self.bm25[False].get_scores(tokens(query, stem=False))
        elif method == "bm25_stem":
            scores = self.bm25[True].get_scores(tokens(query, stem=True))
        elif method == "dense":
            scores = self.vecs @ self.embed_fn([query])[0]
        elif method == "hybrid":
            return rrf([self.rank(query, "bm25_stem"), self.rank(query, "dense")])
        else:
            raise ValueError(method)
        return rank_articles(self.articles, np.asarray(scores))

    def top_chunks(self, query, article, n=2):
        scores = self.vecs @ self.embed_fn([query])[0]
        best = [i for i in np.argsort(-scores, kind="stable") if self.articles[i] == article][:n]
        return [self.chunks[i]["text"] for i in best]
