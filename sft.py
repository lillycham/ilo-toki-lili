"""Supervised fine-tuning on Toki Pona multi-turn Reddit dialogues."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

from tokenizer import TokiPonaTokenizer
from train import pick_device

TEST_PROMPTS = [
    "toki!",
    "sina pilin seme?",
    "sina pali e seme lon tenpo ni?",
    "soweli li pona anu seme?",
    "mi wile sona e toki pona.",
]


def encode_dialogue(
    turns: list[dict[str, str]],
    tokenizer: TokiPonaTokenizer,
    max_len: int,
) -> tuple[list[int], list[int]] | None:
    """Encode a dialogue with loss masked on 'jan:' turns and 'ilo:' prefixes.

    Internal newlines are collapsed to spaces so '\\n' acts cleanly as the
    turn-end delimiter during chat generation.
    """
    input_ids: list[int] = [tokenizer.bos_token_id]
    labels: list[int] = [-100]

    nl_id = tokenizer.tok2id["\n"]
    jan_prefix = tokenizer.encode("jan:")
    ilo_prefix = tokenizer.encode("ilo:")

    for idx, turn in enumerate(turns):
        text = " ".join(turn["text"].split())
        if not text:
            continue
        body_ids = tokenizer.encode(text)
        is_last = idx == len(turns) - 1
        end_id = tokenizer.eos_token_id if is_last else nl_id

        if turn["role"] == "jan":
            seg = jan_prefix + body_ids + [nl_id]
            input_ids.extend(seg)
            labels.extend([-100] * len(seg))
        else:
            input_ids.extend(ilo_prefix)
            labels.extend([-100] * len(ilo_prefix))
            reply = body_ids + [end_id]
            input_ids.extend(reply)
            labels.extend(reply)

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    # Need at least some supervised tokens.
    if not any(lbl != -100 for lbl in labels):
        return None
    return input_ids, labels


def make_batch(
    examples: list[tuple[list[int], list[int]]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x, _ in examples)
    bsz = len(examples)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.long)

    for i, (ids, lbls) in enumerate(examples):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(lbls, dtype=torch.long)
        attn_mask[i, :n] = 1

    return input_ids.to(device), labels.to(device), attn_mask.to(device)


@torch.no_grad()
def evaluate(
    model: LlamaForCausalLM,
    val_examples: list[tuple[list[int], list[int]]],
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for i in range(0, len(val_examples), batch_size):
        batch = val_examples[i : i + batch_size]
        x, y, mask = make_batch(batch, pad_id, device)
        out = model(input_ids=x, attention_mask=mask, labels=y)
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def reply_once(
    model: LlamaForCausalLM,
    tokenizer: TokiPonaTokenizer,
    user_msg: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.75,
) -> str:
    model.eval()
    prompt_ids = (
        [tokenizer.bos_token_id]
        + tokenizer.encode(f"jan: {' '.join(user_msg.split())}\nilo:")
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
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
    gen_ids = out[0][len(prompt_ids) :].tolist()
    model.train()
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="SFT a Toki Pona base model on Reddit dialogues")
    p.add_argument("--base", type=Path, default=Path("checkpoints/14m"))
    p.add_argument("--data", type=Path, default=Path("data/reddit_dialogues.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/14m-chat"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"Using device: {device}")

    tokenizer = TokiPonaTokenizer()
    raw_dialogues = [
        json.loads(line)["turns"]
        for line in args.data.read_text().splitlines()
        if line.strip()
    ]
    examples = []
    for turns in raw_dialogues:
        enc = encode_dialogue(turns, tokenizer, max_len=args.max_len)
        if enc is not None:
            examples.append(enc)

    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.95))
    train_ex = examples[:split]
    val_ex = examples[split:]
    print(f"Loaded {len(train_ex):,} train and {len(val_ex):,} val dialogues")

    model = LlamaForCausalLM.from_pretrained(args.base).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05
    )

    steps_per_epoch = math.ceil(len(train_ex) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    val_loss = evaluate(model, val_ex, args.batch_size, tokenizer.pad_token_id, device)
    print(f"Initial val_loss: {val_loss:.3f} (ppl {math.exp(min(val_loss, 20)):.2f})")

    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_ex)
        model.train()
        running = 0.0
        n_batches = 0

        for i in range(0, len(train_ex), args.batch_size):
            batch = train_ex[i : i + args.batch_size]
            x, y, mask = make_batch(batch, tokenizer.pad_token_id, device)
            out = model(input_ids=x, attention_mask=mask, labels=y)
            loss = out.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            n_batches += 1
            global_step += 1

        val_loss = evaluate(model, val_ex, args.batch_size, tokenizer.pad_token_id, device)
        ppl = math.exp(min(val_loss, 20.0))
        print(
            f"epoch {epoch}/{args.epochs} (step {global_step}, {time.time() - t0:.1f}s) | "
            f"train_loss {running / max(1, n_batches):.3f} | "
            f"val_loss {val_loss:.3f} | val_ppl {ppl:.2f}"
        )
        for prompt in TEST_PROMPTS:
            ans = reply_once(model, tokenizer, prompt, device)
            print(f"  jan: {prompt}\n  ilo: {ans}")
        print("---")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save(args.out_dir / "tokenizer.json")
    print(f"Saved SFT checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
