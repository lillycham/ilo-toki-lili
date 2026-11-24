"""Fetch and pack Toki Pona text corpora into cached token tensors."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import torch
from datasets import load_dataset

from tokenizer import TokiPonaTokenizer

# Dataset specs: (hf_name, candidate_text_columns)
DATASETS: dict[str, tuple[str, list[str]]] = {
    "poki-lapo": ("NetherQuartz/poki-lapo", ["text", "content", "toki_pona"]),
    "wikipesija": ("NetherQuartz/wikipesija", ["text", "content", "article"]),
    "tatoeba": ("NetherQuartz/tatoeba-tokipona", ["tok", "toki_pona", "text", "sentence"]),
    "monolingual": ("NetherQuartz/tokipona-monolingual", ["text"]),
    "lipu-kule": ("NetherQuartz/lipu-kule", ["text"]),
    "lipu-sewi": ("NetherQuartz/lipu-sewi", ["tok"]),
    "sentences": ("finnnnnnnnnnnn/toki-pona-sentences", ["text", "sentence", "tok"]),
}


def _extract_texts(name: str, hf_name: str, candidate_cols: list[str]) -> list[str]:
    ds = load_dataset(hf_name)

    # Group Bible verses into full chapters for long-form narrative prose.
    if name == "lipu-sewi":
        chapters: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
        for split_name in ds.keys():
            for row in ds[split_name]:
                txt = row.get("tok")
                if isinstance(txt, str) and txt.strip():
                    book = str(row.get("book", ""))
                    ch = int(row.get("chapter") or 0)
                    v = int(row.get("verse") or 0)
                    chapters[(book, ch)].append((v, txt.strip()))
        docs = []
        for _, verses in sorted(chapters.items()):
            verses.sort(key=lambda x: x[0])
            doc = " ".join(t for _, t in verses)
            if len(doc) >= 10:
                docs.append(doc)
        return docs

    texts: list[str] = []
    for split_name in ds.keys():
        split = ds[split_name]
        col = next((c for c in candidate_cols if c in split.column_names), None)
        if col is None:
            for c in split.column_names:
                if len(split) > 0 and isinstance(split[0][c], str):
                    col = c
                    break
        if col is None:
            raise ValueError(f"Could not find text column in {hf_name} ({split.column_names})")
        for row in split[col]:
            if isinstance(row, str):
                cleaned = row.strip()
                if len(cleaned) >= 10:
                    texts.append(cleaned)
    return texts


def load_or_build_corpus(
    dataset_names: list[str],
    tokenizer: TokiPonaTokenizer,
    cache_dir: Path = Path("data"),
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = "+".join(sorted(dataset_names))
    # v2 tag reflects the mid-sentence proper-name tokenizer fix.
    cache_path = cache_dir / f"{key}_v{len(tokenizer)}_v2.pt"

    if cache_path.exists():
        blob = torch.load(cache_path, weights_only=True)
        return blob["train"], blob["val"]

    seen: set[str] = set()
    all_texts: list[str] = []
    for name in dataset_names:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
        hf_name, cols = DATASETS[name]
        print(f"Downloading {hf_name}...")
        texts = _extract_texts(name, hf_name, cols)
        added = 0
        for t in texts:
            if t not in seen:
                seen.add(t)
                all_texts.append(t)
                added += 1
        print(f"  loaded {len(texts):,} docs from {name} ({added:,} unique)")

    rng = random.Random(seed)
    rng.shuffle(all_texts)

    split_idx = max(1, int(len(all_texts) * (1.0 - val_fraction)))
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]

    def pack(docs: list[str]) -> torch.Tensor:
        ids: list[int] = []
        for doc in docs:
            ids.extend(tokenizer.encode(doc, add_bos=True, add_eos=True))
        return torch.tensor(ids, dtype=torch.int32)

    train_ids = pack(train_texts)
    val_ids = pack(val_texts)

    torch.save({"train": train_ids, "val": val_ids}, cache_path)
    print(
        f"Cached {len(train_ids):,} train tokens and {len(val_ids):,} val tokens "
        f"to {cache_path}"
    )
    return train_ids, val_ids
