"""Deterministic hybrid tokenizer for Toki Pona.

Encodes closed-class Toki Pona lexical words as single tokens, punctuation as
single tokens, and capitalized proper names / OOV words as character sequences
prefixed by <name> or <spell> so there are no <unk> tokens on normal text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<name>", "<spell>"]

PUNCT_TOKENS = [".", ",", ":", "?", "!", ";", '"', "(", ")", "-", "\n"]

# pu (120 + ali/kin/namako/oko) + ku suli + su = 139 standard Toki Pona words.
LEXICON = [
    "a", "akesi", "ala", "alasa", "ale", "ali", "anpa", "ante", "anu", "awen",
    "e", "en", "epiku", "esun",
    "ijo", "ike", "ilo", "insa",
    "jaki", "jan", "jasima", "jelo", "jo",
    "kala", "kalama", "kama", "kasi", "ken", "kepeken", "kijetesantakalu",
    "kili", "kin", "kipisi", "kiwen", "ko", "kokosila", "kon", "ku", "kule",
    "kulupu", "kute",
    "la", "lanpan", "lape", "laso", "lawa", "leko", "len", "lete", "li",
    "lili", "linja", "lipu", "loje", "lon", "luka", "lukin", "lupa",
    "ma", "mama", "mani", "meli", "meso", "mi", "mije", "misikeke", "moku",
    "moli", "monsi", "monsuta", "mu", "mun", "musi", "mute",
    "n", "namako", "nanpa", "nasa", "nasin", "nena", "ni", "nimi", "noka",
    "o", "oko", "olin", "ona", "open",
    "pakala", "pali", "palisa", "pan", "pana", "pi", "pilin", "pimeja",
    "pini", "pipi", "poka", "poki", "pona", "pu",
    "sama", "seli", "selo", "seme", "sewi", "sijelo", "sike", "sin", "sina",
    "sinpin", "sitelen", "soko", "sona", "soweli", "su", "suli", "suno",
    "supa", "suwi",
    "tan", "taso", "tawa", "telo", "tenpo", "toki", "tomo", "tonsi", "tu",
    "unpa", "uta", "utala",
    "walo", "wan", "waso", "wawa", "weka", "wile",
]

CHAR_TOKENS = [f"c:{ch}" for ch in "abcdefghijklmnopqrstuvwxyz0123456789"]

# Splits text into newlines, words (alphanumeric), or single punctuation chars.
_TOKEN_RE = re.compile(r"\n|[A-Za-z0-9]+|[.,:?!;\"()\-]")

_NO_SPACE_BEFORE = {".", ",", ":", "?", "!", ";", ")", "\n"}
_NO_SPACE_AFTER = {"(", "\n"}
_SENTENCE_END_PUNCT = {".", "?", "!", "\n", ":"}


class TokiPonaTokenizer:
    def __init__(self) -> None:
        self.vocab: list[str] = SPECIAL_TOKENS + PUNCT_TOKENS + LEXICON + CHAR_TOKENS
        self.tok2id: dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2tok: dict[int, str] = {i: tok for i, tok in enumerate(self.vocab)}
        self.lexicon_set: set[str] = set(LEXICON)
        self.punct_set: set[str] = set(PUNCT_TOKENS)

        self.pad_token_id = self.tok2id["<pad>"]
        self.bos_token_id = self.tok2id["<bos>"]
        self.eos_token_id = self.tok2id["<eos>"]
        self.unk_token_id = self.tok2id["<unk>"]
        self.name_token_id = self.tok2id["<name>"]
        self.spell_token_id = self.tok2id["<spell>"]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)

        at_sentence_start = True
        for piece in _TOKEN_RE.findall(text):
            if piece in self.punct_set:
                ids.append(self.tok2id[piece])
                if piece in _SENTENCE_END_PUNCT:
                    at_sentence_start = True
                continue

            lower = piece.lower()
            is_capitalized = piece[0].isupper()

            # Toki Pona only capitalizes proper names, though messy corpora
            # sometimes capitalize the first word of a sentence. Normalize
            # capitalized lexicon words only at sentence start; mid-sentence
            # (e.g. "jan Lili", "jan Suno") treat them as <name>.
            if not is_capitalized and lower in self.lexicon_set:
                ids.append(self.tok2id[lower])
            elif is_capitalized and (lower not in self.lexicon_set or not at_sentence_start):
                ids.append(self.name_token_id)
                for ch in lower:
                    ids.append(self.tok2id.get(f"c:{ch}", self.unk_token_id))
            elif is_capitalized and lower in self.lexicon_set:
                ids.append(self.tok2id[lower])
            else:
                ids.append(self.spell_token_id)
                for ch in lower:
                    ids.append(self.tok2id.get(f"c:{ch}", self.unk_token_id))

            at_sentence_start = False

        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        out: list[str] = []
        i = 0
        n = len(ids)

        while i < n:
            tok = self.id2tok.get(ids[i], "<unk>")
            if tok in ("<pad>", "<bos>", "<eos>"):
                if not skip_special_tokens:
                    out.append(tok)
                i += 1
                continue

            if tok in ("<name>", "<spell>"):
                is_name = tok == "<name>"
                i += 1
                chars: list[str] = []
                while i < n:
                    ntok = self.id2tok.get(ids[i], "")
                    if ntok.startswith("c:"):
                        chars.append(ntok[2:])
                        i += 1
                    else:
                        break
                word = "".join(chars)
                if is_name and word:
                    word = word[0].upper() + word[1:]
                if word:
                    out.append(word)
                continue

            if tok.startswith("c:"):
                # Stray character token without prefix.
                out.append(tok[2:])
                i += 1
                continue

            out.append(tok)
            i += 1

        # Reconstruct spacing cleanly.
        text_parts: list[str] = []
        in_quotes = False
        for idx, piece in enumerate(out):
            if piece == '"':
                if idx > 0 and not in_quotes and out[idx - 1] not in _NO_SPACE_AFTER:
                    text_parts.append(" ")
                text_parts.append('"')
                in_quotes = not in_quotes
                continue

            if idx > 0:
                prev = out[idx - 1]
                need_space = (
                    piece not in _NO_SPACE_BEFORE
                    and prev not in _NO_SPACE_AFTER
                    and not (prev == '"' and in_quotes)
                )
                if need_space:
                    text_parts.append(" ")
            text_parts.append(piece)

        return "".join(text_parts)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"vocab": self.vocab}, indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    tok = TokiPonaTokenizer()
    print(f"Vocab size: {len(tok)}")
    sample = "toki! mi jan Sonja. mi wile pali e ilo toki lili lon ma Kanse."
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    print("Tokens:", [tok.id2tok[i] for i in ids])
    print("Decoded:", tok.decode(ids))
