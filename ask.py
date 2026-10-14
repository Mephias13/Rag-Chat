import sys

import llm
from kb import load_articles, load_chunks
from retrieval import Retriever

ANSWER_MODEL = "qwen3.5:9b"
TOP_ARTICLES = 3
CHUNKS_PER_ARTICLE = 2
SYSTEM = (
    "Ты помощник чата поддержки банка. Отвечай только по приведённым выдержкам, по-русски, "
    "3-6 предложений. В конце укажи номера использованных источников в квадратных скобках. "
    "Если в выдержках нет ответа, так и скажи и предложи обратиться к оператору. Ничего не выдумывай."
)


def main(question):
    titles = {a["id"]: a["title"] for a in load_articles()}
    retriever = Retriever(load_chunks())
    top = retriever.rank(question, "hybrid")[:TOP_ARTICLES]  # flat search: E3 showed hard routing hurts
    context = "\n\n".join(
        f"[{n}]\n" + "\n".join(retriever.top_chunks(question, a, CHUNKS_PER_ARTICLE))
        for n, a in enumerate(top, 1)
    )
    answer = llm.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Выдержки:\n{context}\n\nВопрос клиента: {question}"}],
        model=ANSWER_MODEL,
    )
    print(f"\n{answer}\n\nИсточники:")
    for n, a in enumerate(top, 1):
        print(f"[{n}] {titles[a]}  https://fincult.info/article/{a}/")
    llm.unload(ANSWER_MODEL)
    llm.unload(llm.EMBED_MODEL)


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
