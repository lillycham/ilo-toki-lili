"""Three-way SFT ablation: reddit-only vs tiny-dialogues-only vs 50/50 mix.

All three arms start from the same pretrained base checkpoint, train for the
same number of gradient steps, and are cross-evaluated on held-out validation
splits from BOTH datasets plus a fixed suite of single- and multi-turn probes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import LlamaForCausalLM

from sft import encode_dialogue, evaluate, make_batch
from tokenizer import TokiPonaTokenizer
from train import pick_device

SINGLE_PROBES = [
    "toki!",
    "mi jan Lili.",
    "sina pilin seme?",
    "toki pona li pona anu seme?",
    "mi wile sona e toki pona.",
]

MULTI_TURN_PROBE = [
    "mi jo e soweli lili. ona li walo.",
    "ona li pona anu seme?",
]


def load_tiny_dialogues(
    out_path: Path,
    turns_per_chunk: int = 4,
) -> list[list[dict[str, str]]]:
    if out_path.exists():
        return [
            json.loads(ln)["turns"]
            for ln in out_path.read_text().splitlines()
            if ln.strip()
        ]

    raw_path = hf_hub_download(
        "Climb-Ai/toki-pona-tiny-dialogues",
        filename="tokipona_dataset.jsonl",
        repo_type="dataset",
    )
    chunks: list[list[dict[str, str]]] = []
    for line in Path(raw_path).read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        raw_msgs = obj.get("messages") or obj.get("conversations") or []
        turns: list[dict[str, str]] = []
        for m in raw_msgs:
            role_raw = (m.get("role") or m.get("from") or "").lower()
            text = (m.get("content") or m.get("value") or "").strip()
            if not text:
                continue
            if role_raw in ("user", "human"):
                role = "jan"
            elif role_raw in ("assistant", "gpt"):
                role = "ilo"
            else:
                continue
            # Ensure alternating jan / ilo.
            if turns and turns[-1]["role"] == role:
                turns[-1]["text"] += " " + text
            else:
                turns.append({"role": role, "text": text})

        # Trim leading ilo or trailing jan so chunks align cleanly.
        while turns and turns[0]["role"] != "jan":
            turns.pop(0)
        while turns and turns[-1]["role"] != "ilo":
            turns.pop()

        for i in range(0, len(turns), turns_per_chunk):
            window = turns[i : i + turns_per_chunk]
            if len(window) >= 2 and window[0]["role"] == "jan" and window[-1]["role"] == "ilo":
                chunks.append(window)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for c in chunks:
            f.write(json.dumps({"turns": c}, ensure_ascii=False) + "\n")
    return chunks


@torch.no_grad()
def chat_generate(
    model: LlamaForCausalLM,
    tokenizer: TokiPonaTokenizer,
    history: list[tuple[str, str]],
    device: torch.device,
    max_new_tokens: int = 48,
    temperature: float = 0.7,
) -> str:
    model.eval()
    ids = [tokenizer.bos_token_id]
    for role, text in history:
        ids.extend(tokenizer.encode(f"{role}: {' '.join(text.split())}\n"))
    ids.extend(tokenizer.encode("ilo:"))
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    nl_id = tokenizer.tok2id["\n"]
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[tokenizer.eos_token_id, nl_id],
    )
    gen_ids = out[0][len(ids) :].tolist()
    model.train()
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def run_arm(
    name: str,
    base_dir: Path,
    train_examples: list[tuple[list[int], list[int]]],
    val_reddit: list[tuple[list[int], list[int]]],
    val_tiny: list[tuple[list[int], list[int]]],
    tokenizer: TokiPonaTokenizer,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    out_dir: Path,
    seed: int = 42,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)

    model = LlamaForCausalLM.from_pretrained(base_dir).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05
    )
    warmup = max(1, steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t0 = time.time()
    model.train()
    idx_pool: list[int] = []
    running = 0.0

    for step in range(1, steps + 1):
        if len(idx_pool) < batch_size:
            more = list(range(len(train_examples)))
            random.shuffle(more)
            idx_pool.extend(more)
        batch = [train_examples[idx_pool.pop()] for _ in range(batch_size)]
        x, y, mask = make_batch(batch, tokenizer.pad_token_id, device)
        out = model(input_ids=x, attention_mask=mask, labels=y)
        loss = out.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running += loss.item()

    elapsed = time.time() - t0
    v_red = evaluate(model, val_reddit, batch_size, tokenizer.pad_token_id, device)
    v_tiny = evaluate(model, val_tiny, batch_size, tokenizer.pad_token_id, device)

    # Fixed-seed qualitative probes.
    torch.manual_seed(1234)
    single_out = {}
    for prompt in SINGLE_PROBES:
        single_out[prompt] = chat_generate(model, tokenizer, [("jan", prompt)], device)

    torch.manual_seed(1234)
    hist: list[tuple[str, str]] = []
    multi_out: list[tuple[str, str]] = []
    for u in MULTI_TURN_PROBE:
        hist.append(("jan", u))
        r = chat_generate(model, tokenizer, hist, device)
        hist.append(("ilo", r))
        multi_out.append((u, r))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save(out_dir / "tokenizer.json")

    print(
        f"[{name}] {steps} steps in {elapsed:.1f}s | "
        f"val_reddit={v_red:.3f} (ppl {math.exp(v_red):.2f}) | "
        f"val_tiny={v_tiny:.3f} (ppl {math.exp(v_tiny):.2f})"
    )
    return {
        "name": name,
        "val_reddit_loss": v_red,
        "val_reddit_ppl": math.exp(v_red),
        "val_tiny_loss": v_tiny,
        "val_tiny_ppl": math.exp(v_tiny),
        "single_probes": single_out,
        "multi_probe": multi_out,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("checkpoints/14m"))
    p.add_argument("--reddit", type=Path, default=Path("data/reddit_dialogues.jsonl"))
    p.add_argument("--tiny", type=Path, default=Path("data/tiny_dialogues.jsonl"))
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", default="checkpoints/ablation")
    args = p.parse_args()

    device = pick_device(None)
    tokenizer = TokiPonaTokenizer()

    reddit_raw = [
        json.loads(ln)["turns"]
        for ln in args.reddit.read_text().splitlines()
        if ln.strip()
    ]
    tiny_raw = load_tiny_dialogues(args.tiny, turns_per_chunk=4)

    def encode_all(raw: list[list[dict[str, str]]]) -> list[tuple[list[int], list[int]]]:
        out = []
        for turns in raw:
            enc = encode_dialogue(turns, tokenizer, max_len=args.max_len)
            if enc is not None:
                out.append(enc)
        return out

    red_enc = encode_all(reddit_raw)
    tiny_enc = encode_all(tiny_raw)

    rng = random.Random(args.seed)
    rng.shuffle(red_enc)
    rng.shuffle(tiny_enc)

    r_split = int(len(red_enc) * 0.9)
    t_split = int(len(tiny_enc) * 0.9)

    red_train, red_val = red_enc[:r_split], red_enc[r_split:]
    tiny_train, tiny_val = tiny_enc[:t_split], tiny_enc[t_split:]

    # Balanced 50/50 mix: upsample smaller train set to match larger.
    n_max = max(len(red_train), len(tiny_train))
    mix_red = [red_train[i % len(red_train)] for i in range(n_max)]
    mix_tiny = [tiny_train[i % len(tiny_train)] for i in range(n_max)]
    mix_train = mix_red + mix_tiny
    rng.shuffle(mix_train)

    print(
        f"Reddit: {len(red_train):,} train / {len(red_val):,} val | "
        f"Tiny-dialogues: {len(tiny_train):,} train / {len(tiny_val):,} val | "
        f"Mix: {len(mix_train):,} balanced train"
    )

    # Baseline (base model before SFT).
    base_model = LlamaForCausalLM.from_pretrained(args.base).to(device)
    b_red = evaluate(base_model, red_val, args.batch_size, tokenizer.pad_token_id, device)
    b_tiny = evaluate(base_model, tiny_val, args.batch_size, tokenizer.pad_token_id, device)
    del base_model
    print(
        f"[base]   0 steps | "
        f"val_reddit={b_red:.3f} (ppl {math.exp(b_red):.2f}) | "
        f"val_tiny={b_tiny:.3f} (ppl {math.exp(b_tiny):.2f})"
    )

    results = []
    for name, tr, out_dir in [
        ("reddit", red_train, Path(f"{args.out_prefix}-reddit")),
        ("tiny", tiny_train, Path(f"{args.out_prefix}-tiny")),
        ("mix", mix_train, Path(f"{args.out_prefix}-mix")),
    ]:
        res = run_arm(
            name=name,
            base_dir=args.base,
            train_examples=tr,
            val_reddit=red_val,
            val_tiny=tiny_val,
            tokenizer=tokenizer,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            out_dir=out_dir,
            seed=args.seed,
        )
        results.append(res)

    print("\n=== PROBE COMPARISON ===")
    for prompt in SINGLE_PROBES:
        print(f"\njan: {prompt}")
        for r in results:
            print(f"  [{r['name']:6s}] ilo: {r['single_probes'][prompt]}")

    print("\n=== MULTI-TURN REFERENT PROBE ===")
    for r in results:
        print(f"\n[{r['name']}]")
        for u, a in r["multi_probe"]:
            print(f"  jan: {u}")
            print(f"  ilo: {a}")


if __name__ == "__main__":
    main()
