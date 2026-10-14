import html
import re
import time
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API = "https://fincult.info/api/v1/articles/item/"  # the site is a SPA, plain HTML has no text
HEADERS = {"User-Agent": "support-rag-eval research script"}
URLS = Path("kb_urls.tsv")
OUT = Path("data/kb")
HEADING = {"h2": "##", "h3": "###"}


def clean(fragment):
    soup = BeautifulSoup(html.unescape(fragment or ""), "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert(0, "- ")
    for el in soup.find_all(["p", "li", "div", "tr", "h1", "h2", "h3", "h4", "ul", "ol", "table"]):
        el.append("\n")
    text = soup.get_text("").replace("\xa0", " ").replace("­", "")
    lines = (re.sub(r"[ \t\r]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def to_markdown(doc, skipped):
    parts = [f"# {clean(doc['name'])}"]
    intro = clean(doc.get("description"))
    if intro:
        parts.append(intro)
    for block in doc.get("content", []):
        data = block.get("data", {})
        title, text = clean(data.get("title")), clean(data.get("text"))
        if not (title or text):
            skipped[block.get("code")] += 1
            continue
        if title:
            parts.append(f"{HEADING.get(data.get('titleType'), '##')} {title}")
        if text:
            parts.append(text)
    return "\n".join(parts) + "\n"


def fetch(slug):
    for attempt in range(3):
        try:
            resp = requests.get(API + slug, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    rows = [line.split("\t") for line in URLS.read_text(encoding="utf-8").splitlines() if line.strip()]
    skipped, lengths, out_rows = Counter(), [], []
    for topic, url, *_ in rows:
        slug = url.rstrip("/").rsplit("/", 1)[1]
        path = OUT / topic / f"{slug}.md"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(to_markdown(fetch(slug), skipped), encoding="utf-8")
            time.sleep(0.5)
        text = path.read_text(encoding="utf-8")
        lengths.append(len(text))
        out_rows.append([topic, url, text.splitlines()[0].removeprefix("# ")])
    URLS.write_text("".join("\t".join(r) + "\n" for r in out_rows), encoding="utf-8", newline="\n")
    lengths.sort()
    print(f"{len(lengths)} articles, chars min/median/max = {lengths[0]}/{lengths[len(lengths)//2]}/{lengths[-1]}")
    print("blocks without text or title:", dict(skipped))


if __name__ == "__main__":
    main()
