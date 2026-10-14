from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

import llm

ROUTER_MODEL = "qwen3.5:4b"
TOPICS = {
    "refunds": "списания и возврат денег, споры с продавцом или банком, навязанные услуги и страховки, права потребителя",
    "credits": "кредиты и займы, долги, коллекторы, банкротство, кредитная история, поручительство",
    "cards": "банковские карты, их блокировка и потеря, блокировка счёта, овердрафт, кешбэк, интернет-банк",
    "payments": "переводы и платежи, реквизиты, QR-коды, счета и вклады, ошибочные переводы",
    "fraud": "мошенничество, кража денег и данных, фишинг, безопасность",
}

PROMPT = (
    "Определи тему обращения клиента в чат банка. Темы:\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in TOPICS.items())
    + "\nОтветь одним словом: идентификатором темы."
)


def route_llm(question, model=ROUTER_MODEL):
    answer = llm.chat(
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": question}],
        model=model, keep_alive="10m",
    ).lower()
    return next((t for t in TOPICS if t in answer), "unknown")


def tfidf_pipeline():
    # char n-grams cope with Russian morphology without a stemmer
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )


def route_tfidf(texts, labels, seed=0):
    """Out-of-fold predictions, so no question is routed by a model that saw it in training."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cross_val_predict(tfidf_pipeline(), texts, labels, cv=cv))
