import json
from pathlib import Path

import llm
from kb import load_articles, tokens

MODEL = "qwen3.5:9b"
OUT = Path("questions.jsonl")
MAX_OVERLAP = 0.6  # share of question words that also sit in the title; higher = too easy for BM25
ATTEMPTS = 3

SYSTEM = (
    "Ты клиент банка и пишешь сообщение в чат поддержки в мобильном приложении. "
    "Тебе дают статью, на которую должен ответить оператор. Напиши ОДНО сообщение клиента, "
    "на которое эта статья отвечает. Правила: 1-2 коротких предложения, не длиннее 200 символов, разговорный стиль, как пишут "
    "в чате, без приветствий. Опиши свою ситуацию или проблему своими словами и задай конкретный вопрос именно по теме статьи. НЕ повторяй заголовок "
    "статьи и её термины дословно, не упоминай слово «статья». Ответь только текстом сообщения."
)
RETRY = "Прошлый вариант слишком похож на заголовок. Опиши ту же ситуацию совсем другими словами."


def overlap(question, title):
    q = {w for w in tokens(question) if len(w) >= 4}
    t = set(tokens(title))
    return len(q & t) / len(q) if q else 1.0


def generate(article):
    best = None
    for attempt in range(ATTEMPTS):
        user = f"Заголовок: {article['title']}\n\n{article['body'][:3500]}"
        if attempt:
            user += f"\n\n{RETRY}"
        text = llm.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            model=MODEL, temperature=0.5,
        )
        text = text.strip().strip('«»"').splitlines()[0].strip() if text.strip() else ""
        score = overlap(text, article["title"])
        if text and (best is None or score < best[1]):
            best = (text, score)
        if best and best[1] <= MAX_OVERLAP:
            break
    return best


def main():
    done = set()
    if OUT.exists():
        done = {json.loads(line)["gold"] for line in OUT.read_text(encoding="utf-8").splitlines()}
    articles = load_articles()
    with OUT.open("a", encoding="utf-8") as f:
        for i, article in enumerate(articles, 1):
            if article["id"] in done:
                continue
            text, score = generate(article)
            row = {"id": f"q{i:03d}", "text": text, "gold": article["id"], "topic": article["topic"],
                   "overlap": round(score, 2), "source": "llm"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"{i}/{len(articles)}", flush=True)
    llm.unload(MODEL)


if __name__ == "__main__":
    main()
