"""3x2 depth/looping ablation with WSD schedule:
  Rows:    unlooped-4L (~7M, 4L) | looped-4Lx3 (~7M, 12L) | unlooped-12L (~21M, 12L)
  Columns: standard residual     | depth attention (AttnRes)

Evaluates on overall validation, long-form prose validation (poki-lapo +
lipu-kule + lipu-sewi + wikipesija), and short Tatoeba validation, and logs
metrics + best checkpoints per arm.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path

import torch

from arch import TokiLlama, TokiLlamaConfig
from data import DATASETS, _extract_texts
from tokenizer import TokiPonaTokenizer
from train import pick_device, sample_batch

ARMS: dict[str, tuple[int, int, bool]] = {
    # name: (num_unique_layers, num_loops, use_depth_attn)
    "u4-std": (4, 1, False),
    "u4-da": (4, 1, True),
    "l4x3-std": (4, 3, False),
    "l4x3-da": (4, 3, True),
    "u12-std": (12, 1, False),
    "u12-da": (12, 1, True),
}

FULL_DATASETS = ["poki-lapo", "wikipesija", "tatoeba", "monolingual", "lipu-kule", "lipu-sewi"]
LONGFORM_SET = {"poki-lapo", "wikipesija", "lipu-kule", "lipu-sewi"}


def load_leakfree_splits(
    tokenizer: TokiPonaTokenizer,
    cache_dir: Path = Path("data"),
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build train, val_all, and val_long (strict subset of val_all, zero train overlap)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"depth_ablation_splits_v{len(tokenizer)}_s{seed}.pt"
    if cache_path.exists():
        blob = torch.load(cache_path, weights_only=True)
        return blob["train"], blob["val_all"], blob["val_long"]

    import random
    rng = random.Random(seed)
    seen: set[str] = set()
    train_docs: list[str] = []
    val_all_docs: list[str] = []
    val_long_docs: list[str] = []

    for name in FULL_DATASETS:
        hf_name, cols = DATASETS[name]
        texts = _extract_texts(name, hf_name, cols)
        uniq = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        rng.shuffle(uniq)
        split = max(1, int(len(uniq) * (1.0 - val_fraction)))
        tr, va = uniq[:split], uniq[split:]
        train_docs.extend(tr)
        val_all_docs.extend(va)
        if name in LONGFORM_SET:
            val_long_docs.extend(va)

    rng.shuffle(train_docs)
    rng.shuffle(val_all_docs)
    rng.shuffle(val_long_docs)

    def pack(docs: list[str]) -> torch.Tensor:
        ids: list[int] = []
        for d in docs:
            ids.extend(tokenizer.encode(d, add_bos=True, add_eos=True))
        return torch.tensor(ids, dtype=torch.int32)

    train_t = pack(train_docs)
    val_all_t = pack(val_all_docs)
    val_long_t = pack(val_long_docs)
    torch.save({"train": train_t, "val_all": val_all_t, "val_long": val_long_t}, cache_path)
    return train_t, val_all_t, val_long_t


@torch.no_grad()
def evaluate_deterministic(
    model: TokiLlama,
    val_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    max_batches: int = 32,
) -> float:
    """Deterministic contiguous-slice validation loss (zero sampling variance)."""
    model.eval()
    stride = batch_size * seq_len
    n_batches = min(max_batches, (len(val_tokens) - 1) // stride)
    losses = []
    for b in range(n_batches):
        start = b * stride
        chunk = val_tokens[start : start + stride].view(batch_size, seq_len)
        x = chunk.to(device=device, dtype=torch.long)
        _, loss = model(input_ids=x, labels=x)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def run_arm(
    arm_name: str,
    train_tokens: torch.Tensor,
    val_all: torch.Tensor,
    val_long: torch.Tensor,
    tokenizer: TokiPonaTokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    n_unique, n_loops, use_da = ARMS[arm_name]
    torch.manual_seed(args.seed)

    padded_vocab = ((len(tokenizer) + 63) // 64) * 64
    intermediate = ((args.d_model * 8 // 3 + 63) // 64) * 64
    cfg = TokiLlamaConfig(
        vocab_size=padded_vocab,
        hidden_size=args.d_model,
        intermediate_size=intermediate,
        num_unique_layers=n_unique,
        num_loops=n_loops,
        num_attention_heads=args.n_heads,
        max_position_embeddings=args.seq_len,
        use_depth_attn=use_da,
        attention_dropout=args.dropout,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = TokiLlama(cfg).to(device)
    if args.compile and device.type == "cuda":
        model = torch.compile(model)  # type: ignore[assignment]

    raw_model: TokiLlama = model._orig_mod if hasattr(model, "_orig_mod") else model  # type: ignore[assignment]
    total_params = sum(p.numel() for p in raw_model.parameters())
    embed_params = raw_model.embed_tokens.weight.numel()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    decay_start = max(args.warmup_steps, args.steps - args.decay_steps)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, args.steps - decay_start)
        return max(0.0, 1.0 - math.sqrt(min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.bf16
        else nullcontext()
    )

    out_dir = args.out_root / arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, list] = {"train": [], "val_all": [], "val_long": []}

    print(
        f"\n=== [{arm_name}] unique={n_unique} loops={n_loops} eff={n_unique*n_loops} "
        f"depth_attn={use_da} | params={total_params:,} ({total_params - embed_params:,} non-emb) ==="
    )

    train_time = 0.0
    running_loss = 0.0
    best_val = float("inf")
    best_long = float("inf")
    log_interval = max(1, min(100, args.eval_every))

    model.train()
    for step in range(1, args.steps + 1):
        t0 = time.time()
        x = sample_batch(train_tokens, args.batch_size, args.seq_len, device)
        with autocast_ctx:
            _, loss = model(input_ids=x, labels=x)
            assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        train_time += time.time() - t0

        if step % log_interval == 0 or step == args.steps:
            denom = log_interval if step % log_interval == 0 else (step % log_interval)
            avg_loss = running_loss / denom
            running_loss = 0.0
            tok_s = (step * args.batch_size * args.seq_len) / max(train_time, 1e-6)
            lr_now = scheduler.get_last_lr()[0]
            metrics["train"].append([step, round(avg_loss, 4), lr_now])
            print(
                f"  step {step:4d}/{args.steps} | tr {avg_loss:.3f} | "
                f"lr {lr_now:.2e} | {tok_s:,.0f} tok/s"
            )

        in_decay = step > decay_start
        do_eval = (
            step % args.eval_every == 0
            or (in_decay and step % args.decay_eval_every == 0)
            or step == args.steps
        )
        if do_eval:
            v_all = evaluate_deterministic(raw_model, val_all, args.batch_size, args.seq_len, device)
            v_long = evaluate_deterministic(raw_model, val_long, args.batch_size, args.seq_len, device)
            metrics["val_all"].append([step, round(v_all, 4)])
            metrics["val_long"].append([step, round(v_long, 4)])
            tag = ""
            if v_all < best_val:
                best_val = v_all
                best_long = v_long
                raw_model.save_pretrained(out_dir)
                tokenizer.save(out_dir / "tokenizer.json")
                tag = " [best]"
            print(
                f"  -> eval @{step}: val_all={v_all:.3f} (ppl {math.exp(v_all):.2f}) | "
                f"val_long={v_long:.3f} (ppl {math.exp(v_long):.2f}){tag}"
            )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {
        "arm": arm_name,
        "params": total_params,
        "eff_layers": n_unique * n_loops,
        "depth_attn": use_da,
        "best_val_all": best_val,
        "best_ppl_all": math.exp(best_val),
        "best_val_long": best_long,
        "best_ppl_long": math.exp(best_long),
        "train_sec": round(train_time, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="3x2 Looped x Depth-Attention WSD ablation")
    p.add_argument("--arms", default=",".join(ARMS.keys()), help="Comma-separated arms to run")
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--decay-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--decay-eval-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile on CUDA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--out-root", type=Path, default=Path("checkpoints/depth-ablation"))
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device} | bf16={args.bf16} | compile={args.compile}")
    tokenizer = TokiPonaTokenizer()

    train_tokens, val_all, val_long = load_leakfree_splits(tokenizer, seed=args.seed)
    print(
        f"Tokens: train={len(train_tokens):,} | val_all={len(val_all):,} | "
        f"val_long={len(val_long):,}"
    )

    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    summary = []
    for arm in arm_list:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm '{arm}'. Choose from: {list(ARMS)}")
        summary.append(run_arm(arm, train_tokens, val_all, val_long, tokenizer, args, device))

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ 3x2 ABLATION SUMMARY ================")
    print(f"{'Arm':10s} | {'Params':>10s} | {'EffL':>4s} | {'ValAll(ppl)':>14s} | {'ValLong(ppl)':>14s} | {'Sec':>6s}")
    print("-" * 74)
    for r in summary:
        print(
            f"{r['arm']:10s} | {r['params']:10,d} | {r['eff_layers']:4d} | "
            f"{r['best_val_all']:.3f} ({r['best_ppl_all']:5.2f}) | "
            f"{r['best_val_long']:.3f} ({r['best_ppl_long']:5.2f}) | {r['train_sec']:6.1f}"
        )


if __name__ == "__main__":
    main()
