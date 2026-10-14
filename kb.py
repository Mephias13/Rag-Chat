import re
from pathlib import Path

import snowballstemmer

KB_DIR = Path("data/kb")
MAX_CHARS = 800

_stemmer = snowballstemmer.stemmer("russian")
_WORD = re.compile(r"[a-zа-я0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def tokens(text, stem=True):
    words = _WORD.findall(text.lower().replace("ё", "е"))
    return _stemmer.stemWords(words) if stem else words


def load_articles(kb_dir=KB_DIR):
    articles = []
    for path in sorted(Path(kb_dir).rglob("*.md")):
        title, _, body = path.read_text(encoding="utf-8").partition("\n")
        articles.append({"id": path.stem, "topic": path.parent.name, "title": title.removeprefix("# "), "body": body})
    return articles


def _split(line, limit):
    if len(line) <= limit:
        return [line]
    pieces, cur = [], ""
    for sent in _SENTENCE.split(line):
        while len(sent) > limit:  # no sentence boundary, cut hard
            pieces.append(sent[:limit])
            sent = sent[limit:]
        if cur and len(cur) + 1 + len(sent) > limit:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}" if cur else sent
    if cur:
        pieces.append(cur)
    return pieces


def chunk(title, body, limit=MAX_CHARS):
    sections, head, lines = [], "", []
    for line in body.splitlines():
        if line.startswith("#"):
            sections.append((head, lines))
            head, lines = line.lstrip("# ").strip(), []
        else:
            lines.append(line)
    sections.append((head, lines))

    chunks = []
    for head, lines in sections:
        prefix = f"{title}\n{head}\n" if head else f"{title}\n"
        budget = limit - len(prefix)
        cur = ""
        for piece in (p for line in lines for p in _split(line, budget)):
            if cur and len(cur) + 1 + len(piece) > budget:
                chunks.append(prefix + cur)
                cur = piece
            else:
                cur = f"{cur}\n{piece}" if cur else piece
        if cur:
            chunks.append(prefix + cur)
    return chunks


def load_chunks(kb_dir=KB_DIR):
    return [
        {"article": a["id"], "topic": a["topic"], "text": text}
        for a in load_articles(kb_dir)
        for text in chunk(a["title"], a["body"])
    ]
