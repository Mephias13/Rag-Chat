from kb import MAX_CHARS, chunk, tokens


def test_stemming_unifies_forms():
    assert tokens("подписки") == tokens("подписку")
    assert tokens("подписки", stem=False) != tokens("подписку", stem=False)


def test_yo_is_normalized():
    assert tokens("всё") == tokens("все")


def test_chunks_keep_title_and_respect_limit():
    body = "Введение.\n## Раздел\n" + "\n".join(f"Предложение номер {i} про карты и платежи. " * 3 for i in range(40))
    chunks = chunk("Заголовок статьи", body)
    assert len(chunks) > 2
    assert all(c.startswith("Заголовок статьи\n") for c in chunks)
    assert all(len(c) <= MAX_CHARS for c in chunks)


def test_long_line_without_sentence_boundary_is_cut():
    chunks = chunk("T", "x" * 3000)
    assert all(len(c) <= MAX_CHARS for c in chunks)
    assert sum(len(c) - len("T\n") for c in chunks) == 3000


import numpy as np

from retrieval import Retriever, rank_articles, rrf

TOY = [
    {"article": "sub", "topic": "x", "text": "Отмена подписки и автоплатежей. Отключите автопродление."},
    {"article": "card", "topic": "x", "text": "Блокировка карты. Позвоните в банк, чтобы заблокировать карту."},
    {"article": "credit", "topic": "y", "text": "Досрочное погашение кредита."},
]


def fake_embed(texts):
    vecs = np.array([[t.lower().count(w) for w in ("подпис", "карт", "кредит")] for t in texts], dtype=float)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


def test_rrf_prefers_items_ranked_high_in_both_lists():
    assert rrf([["a", "b", "c"], ["b", "a", "c"], ["b", "c", "a"]])[0] == "b"


def test_article_is_scored_by_its_best_chunk():
    assert rank_articles(["a", "a", "b"], np.array([0.1, 0.9, 0.5])) == ["a", "b"]


def test_stemming_fixes_bm25_on_russian_morphology():
    r = Retriever(TOY, embed_fn=fake_embed)
    assert r.rank("отключить подписку", "bm25_stem")[0] == "sub"
    scores_raw = r.bm25[False].get_scores(["подписку"])
    scores_stem = r.bm25[True].get_scores(["подписк"])
    assert scores_raw.max() == 0 and scores_stem.max() > 0


def test_dense_and_hybrid_find_the_right_article():
    r = Retriever(TOY, embed_fn=fake_embed)
    assert r.rank("подписку", "dense")[0] == "sub"
    assert r.rank("подписку", "hybrid")[0] == "sub"


import math

from evaluate import boot_means, metrics, stat, stat_delta


def test_metrics_known_values():
    assert metrics(["a", "b", "c"], "a") == (1.0, 1.0, 1.0)
    assert metrics(["x", "y", "a"], "a") == (0.0, 1.0, 1 / 3)
    assert metrics(["x"] * 6 + ["a"], "a") == (0.0, 0.0, 1 / 7)
    assert metrics(["x"], "a") == (0.0, 0.0, 0.0)
    assert metrics(["x"] * 10 + ["a"], "a")[2] == 0.0


def test_bootstrap_interval_covers_the_mean_and_is_reproducible():
    values = [1.0] * 30 + [0.0] * 20
    s = stat(values)
    assert s["lo"] <= s["mean"] <= s["hi"] and s["mean"] == 0.6
    assert s == stat(values)
    assert np.allclose(boot_means(values), boot_means(values))


def test_paired_delta_sign_and_degenerate_case():
    assert stat_delta([1, 1, 1, 0], [0, 0, 1, 0])["lo"] >= 0
    zero = stat_delta([1, 0, 1], [1, 0, 1])
    assert zero == {"mean": 0.0, "lo": 0.0, "hi": 0.0}


def test_top_chunks_stay_inside_the_article():
    chunks = TOY + [{"article": "sub", "topic": "x", "text": "Подписку можно отключить в приложении."}]
    r = Retriever(chunks, embed_fn=fake_embed)
    got = r.top_chunks("подписку", "sub", n=2)
    assert len(got) == 2 and all("подпис" in t.lower() for t in got)
    assert r.top_chunks("подписку", "card", n=5) == [TOY[1]["text"]]
