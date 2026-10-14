import json
import math
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

import llm
from kb import load_chunks
from retrieval import METHODS, Retriever
from router import ROUTER_MODEL, TOPICS, route_llm, route_tfidf, tfidf_pipeline

QUESTIONS = Path("questions.jsonl")
ROUTER_CACHE = Path("data/cache/router_llm.json")
SEED = 0
N_BOOT = 2000
SIZES = (40, 70, 100)  # KB sizes in articles for E3
LABELS = list(TOPICS)


def metrics(ranking, gold):
    """hit@1, hit@5, reciprocal rank (cut at 10) of the gold article in a ranked list."""
    pos = ranking.index(gold) + 1 if gold in ranking else math.inf
    return float(pos <= 1), float(pos <= 5), 1.0 / pos if pos <= 10 else 0.0


def boot_indices(n):
    # same seed and n give the same resamples, so differences between methods are paired
    return np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))


def boot_means(values):
    values = np.asarray(values, dtype=float)
    return values[boot_indices(len(values))].mean(axis=1)


def stat(values, samples=None):
    values = np.asarray(values, dtype=float)
    samples = boot_means(values) if samples is None else samples
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"mean": round(float(values.mean()), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}


def stat_delta(a, b):
    return stat(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def show(s):
    return f"{s['mean']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]"


def table(header, rows):
    sep = "|".join("---" for _ in header)
    return "\n".join([f"| {' | '.join(header)} |", f"|{sep}|", *(f"| {' | '.join(map(str, r))} |" for r in rows)])


def llm_router_predictions(questions):
    cache = json.loads(ROUTER_CACHE.read_text(encoding="utf-8")) if ROUTER_CACHE.exists() else {}
    todo = [q for q in questions if q["text"] not in cache]
    if todo:
        route_llm("проверка связи")  # model load must not count as latency
        if not llm.on_gpu(ROUTER_MODEL):
            raise RuntimeError(f"{ROUTER_MODEL} is not fully on GPU, restart ollama before timing")
        for q in todo:
            start = time.perf_counter()
            label = route_llm(q["text"])
            cache[q["text"]] = {"label": label, "ms": (time.perf_counter() - start) * 1000}
        ROUTER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ROUTER_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        llm.unload(ROUTER_MODEL)
    return [cache[q["text"]]["label"] for q in questions], [cache[q["text"]]["ms"] for q in questions]


def tfidf_latency(texts, labels):
    pipe = tfidf_pipeline().fit(texts, labels)
    times = []
    for t in texts:
        start = time.perf_counter()
        pipe.predict([t])
        times.append((time.perf_counter() - start) * 1000)
    return times


def run_e1(flat, questions):
    per = {
        m: np.array([metrics(flat.rank(q["text"], m), q["gold"]) for q in questions]) for m in METHODS
    }
    rows = [[m, show(stat(per[m][:, 0])), show(stat(per[m][:, 1])), show(stat(per[m][:, 2]))] for m in METHODS]
    pairs = [("hybrid", "dense"), ("hybrid", "bm25_stem"), ("bm25_stem", "bm25_raw")]
    diffs = [[f"{a} - {b}", show(stat_delta(per[a][:, 1], per[b][:, 1])), show(stat_delta(per[a][:, 2], per[b][:, 2]))]
             for a, b in pairs]
    report = {m: {"hit@1": stat(per[m][:, 0]), "hit@5": stat(per[m][:, 1]), "mrr@10": stat(per[m][:, 2])} for m in METHODS}
    md = table(["method", "hit@1", "hit@5", "MRR@10"], rows)
    md += "\n\nPaired differences (bootstrap over the same resampled questions):\n\n"
    md += table(["pair", "d hit@5", "d MRR@10"], diffs)
    return md, report


def run_e2(questions, preds, latencies):
    y_true = np.array([q["topic"] for q in questions])
    rows, report, matrices = [], {}, []
    for name, pred in preds.items():
        y_pred = np.array(pred)
        f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        f1_boot = [
            f1_score(y_true[i], y_pred[i], labels=LABELS, average="macro", zero_division=0)
            for i in boot_indices(len(y_true))
        ]
        acc, f1s = stat(y_true == y_pred), stat([f1], samples=f1_boot)
        ms = latencies[name]
        report[name] = {"accuracy": acc, "macro_f1": f1s, "ms_p50": round(float(np.percentile(ms, 50)), 1),
                        "ms_p95": round(float(np.percentile(ms, 95)), 1), "unknown": int((y_pred == "unknown").sum())}
        rows.append([name, show(acc), show(f1s), f"{np.percentile(ms, 50):.1f} / {np.percentile(ms, 95):.1f}"])
        cm = confusion_matrix(y_true, y_pred, labels=LABELS + ["unknown"])[: len(LABELS)]
        matrices.append(f"\n{name}, rows = true topic, columns = predicted:\n\n"
                        + table(["true \\ pred", *LABELS, "unknown"], [[t, *row] for t, row in zip(LABELS, cm)]))
    md = table(["router", "accuracy", "macro-F1", "latency p50 / p95, ms"], rows) + "\n" + "\n".join(matrices)
    return md, report


def run_e3(chunks, questions, preds):
    by_topic = {t: sorted({c["article"] for c in chunks if c["topic"] == t}) for t in LABELS}
    rng = random.Random(SEED)
    order = {t: rng.sample(ids, len(ids)) for t, ids in by_topic.items()}  # nested subsets across sizes
    variants = ["flat", "router LLM", "router TF-IDF", "oracle topic"]
    rows, report = [], {}
    for size in SIZES:
        keep = {a for ids in order.values() for a in ids[: size // len(LABELS)]}
        sub = [c for c in chunks if c["article"] in keep]
        flat = Retriever(sub)
        specialists = {t: Retriever([c for c in sub if c["topic"] == t]) for t in LABELS}
        got = {v: [] for v in variants}
        for i, q in enumerate(questions):
            if q["gold"] not in keep:
                continue

            def score(topic):
                return metrics(specialists.get(topic, flat).rank(q["text"], "hybrid"), q["gold"])

            got["flat"].append(metrics(flat.rank(q["text"], "hybrid"), q["gold"]))
            got["router LLM"].append(score(preds["LLM"][i]))
            got["router TF-IDF"].append(score(preds["TF-IDF"][i]))
            got["oracle topic"].append(score(q["topic"]))
        arr = {v: np.array(got[v]) for v in variants}
        for v in variants:
            hit5, mrr = stat(arr[v][:, 1]), stat(arr[v][:, 2])
            d = "-" if v == "flat" else show(stat_delta(arr[v][:, 1], arr["flat"][:, 1]))
            rows.append([size, v, len(arr[v]), show(hit5), show(mrr), d])
            report[f"{size}/{v}"] = {"n": len(arr[v]), "hit@5": hit5, "mrr@10": mrr}
    md = table(["articles in KB", "search", "questions", "hit@5", "MRR@10", "d hit@5 vs flat"], rows)
    return md, report


def main():
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    chunks = load_chunks()
    texts, topics = [q["text"] for q in questions], [q["topic"] for q in questions]

    llm_pred, llm_ms = llm_router_predictions(questions)
    preds = {"LLM": llm_pred, "TF-IDF": route_tfidf(texts, topics, seed=SEED)}
    latencies = {"LLM": llm_ms, "TF-IDF": tfidf_latency(texts, topics)}

    e1_md, e1 = run_e1(Retriever(chunks), questions)
    e2_md, e2 = run_e2(questions, preds, latencies)
    e3_md, e3 = run_e3(chunks, questions, preds)

    head = (
        f"# Results\n\n{len(questions)} questions, {len({c['article'] for c in chunks})} articles, "
        f"{len(chunks)} chunks. Embeddings: {llm.EMBED_MODEL}. LLM router: {ROUTER_MODEL} (zero-shot). "
        f"Intervals are 95% bootstrap ({N_BOOT} resamples of questions, seed {SEED}).\n"
    )
    md = "\n\n".join([
        head,
        "## E1. Retrieval, whole knowledge base (article level)\n\n" + e1_md,
        "## E2. Topic router\n\n" + e2_md,
        "## E3. Flat search vs routed search as the knowledge base grows (hybrid retrieval)\n\n" + e3_md,
    ]) + "\n"
    Path("results.md").write_text(md, encoding="utf-8")
    Path("results.json").write_text(json.dumps({"e1": e1, "e2": e2, "e3": e3}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(md)
    llm.unload(llm.EMBED_MODEL)


if __name__ == "__main__":
    main()
