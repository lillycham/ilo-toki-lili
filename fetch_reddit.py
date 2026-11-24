"""Fetch r/mi_lon and r/tokipona threads from Arctic Shift and build a
filtered multi-turn Toki Pona dialogue dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tokenizer import LEXICON

API_BASE = "https://arctic-shift.photon-reddit.com/api"
USER_AGENT = "ilo-toki-lili/0.1 (toki pona small LM research)"

# Extra common community/ku-lili words allowed by the language filter.
EXTRA_TP_WORDS = {
    "apeja", "isipin", "kamalawala", "kapesi", "linluwi", "majuna",
    "mijomi", "oke", "pake", "pata", "po", "powe", "soto", "teje",
    "wa", "wuwojiti", "yupekosi", "kiki", "kuntu", "mulapisu", "nimisin",
    "penpo", "polinpin", "san", "taki", "te", "to", "usawi",
}
TP_WORDS = set(LEXICON) | EXTRA_TP_WORDS
TP_LETTERS = set("aeioupktmnlwsj")

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_RE = re.compile(r"https?://\S+")
_WORD_RE = re.compile(r"[A-Za-z]+")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")


def api_get(path: str, params: dict[str, str | int]) -> list[dict]:
    qs = urllib.parse.urlencode(params)
    url = f"{API_BASE}{path}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read())
                return payload.get("data", [])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            wait = 1.5 * (2**attempt)
            print(f"  [retry {attempt + 1}] {e} — sleeping {wait:.1f}s")
            time.sleep(wait)
    return []


def fetch_paginated(
    endpoint: str,
    subreddit: str,
    fields: str,
    max_items: int,
    delay: float = 0.25,
) -> list[dict]:
    items: list[dict] = []
    before: int | None = None
    seen: set[str] = set()

    while len(items) < max_items:
        params: dict[str, str | int] = {
            "subreddit": subreddit,
            "limit": 100,
            "sort": "desc",
            "fields": fields,
        }
        if before is not None:
            params["before"] = before

        batch = api_get(endpoint, params)
        if not batch:
            break

        new_count = 0
        min_ts = before
        for row in batch:
            rid = row.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                items.append(row)
                new_count += 1
            ts = row.get("created_utc")
            if isinstance(ts, (int, float)):
                ts_int = int(ts)
                if min_ts is None or ts_int < min_ts:
                    min_ts = ts_int

        if new_count == 0 or min_ts is None:
            break
        before = min_ts - 1

        if len(items) % 1000 < 100:
            print(f"  r/{subreddit} {endpoint}: {len(items):,} items")
        time.sleep(delay)

    return items[:max_items]


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    # Drop markdown quote lines and strip links/URLs.
    lines = [
        ln for ln in raw.splitlines()
        if not ln.lstrip().startswith(">")
    ]
    text = "\n".join(lines)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("", text)
    text = text.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n", text)
    return text.strip()


def is_toki_pona(text: str, min_ratio: float = 0.85, min_tp_words: int = 2) -> bool:
    if not text or text in ("[deleted]", "[removed]"):
        return False
    words = _WORD_RE.findall(text)
    if len(words) < min_tp_words:
        return False

    tp_lex = 0
    valid = 0
    for w in words:
        lw = w.lower()
        if lw in TP_WORDS:
            tp_lex += 1
            valid += 1
        elif w[0].isupper() and set(lw).issubset(TP_LETTERS):
            # Toki Pona proper name (uses only Toki Pona phonology letters).
            valid += 1

    if tp_lex < min_tp_words:
        return False
    return (valid / len(words)) >= min_ratio


def build_dialogues(
    posts: list[dict],
    comments: list[dict],
    min_turns: int = 2,
    max_turns: int = 8,
) -> list[list[dict[str, str]]]:
    nodes: dict[str, dict] = {}

    for p in posts:
        pid = p.get("id")
        author = p.get("author", "")
        if not pid or author in ("AutoModerator", "[deleted]"):
            continue
        title = clean_text(p.get("title", ""))
        body = clean_text(p.get("selftext", ""))
        # Skip r/mi_lon image-only posts whose entire title is just "mi_lon" / "mi lon".
        if not body and title.lower().replace("_", " ").strip() in ("mi lon", "mi lon a"):
            continue
        full = f"{title}\n{body}".strip() if body and body not in ("[deleted]", "[removed]") else title
        if is_toki_pona(full, min_tp_words=4):
            nodes[f"t3_{pid}"] = {
                "id": f"t3_{pid}",
                "parent_id": None,
                "author": author,
                "text": full,
            }

    for c in comments:
        cid = c.get("id")
        author = c.get("author", "")
        if not cid or author in ("AutoModerator", "[deleted]"):
            continue
        body = clean_text(c.get("body", ""))
        if is_toki_pona(body):
            nodes[f"t1_{cid}"] = {
                "id": f"t1_{cid}",
                "parent_id": c.get("parent_id"),
                "author": author,
                "text": body,
            }

    # Find which nodes have at least one valid Toki Pona child so we only emit
    # maximal root-to-leaf chains.
    has_child: set[str] = set()
    for node in nodes.values():
        pid = node["parent_id"]
        if pid in nodes:
            has_child.add(pid)

    dialogues: list[list[dict[str, str]]] = []
    for nid, node in nodes.items():
        if nid in has_child:
            continue
        chain: list[dict] = []
        cur: dict | None = node
        visited: set[str] = set()
        while cur is not None and cur["id"] not in visited:
            visited.add(cur["id"])
            chain.append(cur)
            pid = cur["parent_id"]
            cur = nodes.get(pid) if pid else None

        chain.reverse()
        if len(chain) < min_turns:
            continue

        # Merge consecutive turns by the same author, and cap length.
        merged: list[dict] = []
        for turn in chain:
            if merged and merged[-1]["author"] == turn["author"]:
                merged[-1]["text"] += "\n" + turn["text"]
            else:
                merged.append({"author": turn["author"], "text": turn["text"]})

        if len(merged) < min_turns:
            continue
        merged = merged[:max_turns]

        # Ensure dialogue ends on an assistant ('ilo') turn by trimming if odd
        # length > 2 or pairing alternating turns.
        if len(merged) % 2 == 1:
            merged = merged[:-1]
        if len(merged) < min_turns:
            continue

        turns = []
        for idx, m in enumerate(merged):
            role = "jan" if idx % 2 == 0 else "ilo"
            turns.append({"role": role, "text": m["text"]})
        dialogues.append(turns)

    return dialogues


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Toki Pona Reddit dialogues via Arctic Shift")
    p.add_argument("--subreddits", default="mi_lon,tokipona")
    p.add_argument("--max-comments", type=int, default=8000, help="Max comments per subreddit")
    p.add_argument("--max-posts", type=int, default=2000, help="Max posts per subreddit")
    p.add_argument("--raw-cache", type=Path, default=Path("data/reddit_raw.json"))
    p.add_argument("--out", type=Path, default=Path("data/reddit_dialogues.jsonl"))
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    args.raw_cache.parent.mkdir(parents=True, exist_ok=True)
    subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]

    if args.raw_cache.exists() and not args.refresh:
        print(f"Loading cached raw Reddit data from {args.raw_cache}...")
        raw = json.loads(args.raw_cache.read_text())
        all_posts = raw["posts"]
        all_comments = raw["comments"]
    else:
        all_posts = []
        all_comments = []
        for sub in subreddits:
            print(f"Fetching r/{sub}...")
            posts = fetch_paginated(
                "/posts/search",
                sub,
                "id,author,title,selftext,created_utc",
                max_items=args.max_posts,
            )
            comments = fetch_paginated(
                "/comments/search",
                sub,
                "id,parent_id,link_id,author,body,created_utc",
                max_items=args.max_comments,
            )
            print(f"  r/{sub}: {len(posts):,} posts, {len(comments):,} comments")
            all_posts.extend(posts)
            all_comments.extend(comments)

        args.raw_cache.write_text(
            json.dumps({"posts": all_posts, "comments": all_comments}, ensure_ascii=False)
        )
        print(f"Saved raw cache to {args.raw_cache}")

    dialogues = build_dialogues(all_posts, all_comments)
    total_turns = sum(len(d) for d in dialogues)
    with args.out.open("w") as f:
        for d in dialogues:
            f.write(json.dumps({"turns": d}, ensure_ascii=False) + "\n")

    print(
        f"Built {len(dialogues):,} multi-turn Toki Pona dialogues "
        f"({total_turns:,} total turns) -> {args.out}"
    )
    print("\nSample dialogues:")
    for d in dialogues[:5]:
        print("---")
        for t in d:
            print(f"{t['role']}: {t['text']}")


if __name__ == "__main__":
    main()
