"""Length-matched name-collapse sweep on Toki Pona models.

Tests whether the repeated-piece copy-circuit failure mode (Glegle / Glegley vs
Gegley) reproduces in our character-fallback Toki Pona Llama models across both
192-wide (smoke) and 384-wide (14m, ablation-mix) checkpoints.
"""

from __future__ import annotations

from pathlib import Path
import statistics

import torch
from transformers import LlamaForCausalLM

from tokenizer import TokiPonaTokenizer
from train import pick_device

# 12 names per group, valid Toki Pona phonology, length-matched.
ABAB_4 = [
    "Lili", "Nana", "Momo", "Kiki", "Toto", "Sasa",
    "Papa", "Wawa", "Lala", "Nini", "Mimi", "Koko",
]
UNIQUE_4 = [
    "Meli", "Suno", "Kasi", "Luka", "Pona", "Telo",
    "Walo", "Noka", "Kule", "Seli", "Moku", "Pali",
]
ABABN_5 = [n + "n" for n in ABAB_4]
UNIQUE_5 = [
    "Sonja", "Kanse", "Molin", "Tepan", "Kulan", "Welin",
    "Sopin", "Lamen", "Pikon", "Tasin", "Mekan", "Mupin",
]


@torch.no_grad()
def logit_margin(
    model: LlamaForCausalLM,
    tok: TokiPonaTokenizer,
    prompt_text: str,
    good_id: int,
    bad_id: int,
    device: torch.device,
) -> float:
    ids = tok.encode(prompt_text, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=x).logits[0, -1]
    return (logits[good_id] - logits[bad_id]).item()


@torch.no_grad()
def sample_name_copy(
    model: LlamaForCausalLM,
    tok: TokiPonaTokenizer,
    prompt_text: str,
    device: torch.device,
    n_samples: int = 8,
    temperature: float = 0.8,
) -> list[str]:
    ids = tok.encode(prompt_text, add_bos=True) + [tok.name_token_id]
    x = torch.tensor([ids] * n_samples, dtype=torch.long, device=device)
    out = model.generate(
        input_ids=x,
        max_new_tokens=16,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    names: list[str] = []
    for row in out:
        gen = row[len(ids) :].tolist()
        chars = []
        for tid in gen:
            t = tok.id2tok.get(tid, "")
            if t.startswith("c:"):
                chars.append(t[2:])
            else:
                break
        w = "".join(chars)
        names.append(w[0].upper() + w[1:] if w else "")
    return names


def eval_checkpoint(ckpt: Path, tok: TokiPonaTokenizer, device: torch.device) -> None:
    model = LlamaForCausalLM.from_pretrained(ckpt).to(device)
    model.eval()
    d = model.config.hidden_size
    L = model.config.num_hidden_layers
    print(f"\n================ {ckpt} (d={d}, L={L}) ================")

    li_id = tok.tok2id["li"]
    n_id = tok.tok2id["c:n"]

    # 1. Teacher-forced margins for 4-char names (end-of-name 'li' vs loop-back first char 'c:A').
    for label, names, polluted in [
        ("4-char UNIQUE (clean)", UNIQUE_4, False),
        ("4-char ABAB   (clean)", ABAB_4, False),
        ("4-char ABAB   (1 slip)", ABAB_4, True),
    ]:
        margins = []
        for name in names:
            mention2 = (name + name[:2].lower()) if polluted else name
            prompt = f"tenpo pini la jan {name} li lon ma ni. jan {mention2} li pona. tenpo ni la jan {name}"
            bad_id = tok.tok2id[f"c:{name[0].lower()}"]
            margins.append(logit_margin(model, tok, prompt, li_id, bad_id, device))
        print(
            f"  {label:24s} | mean logit(li)-logit(c:A) = {statistics.mean(margins):+6.2f} "
            f"(sd {statistics.stdev(margins):.2f})"
        )

    # 2. Teacher-forced margins for 5-char names (closer 'c:n' vs loop-back first char 'c:A').
    for label, names, polluted in [
        ("5-char UNIQUE (clean)", UNIQUE_5, False),
        ("5-char ABABn  (clean)", ABABN_5, False),
        ("5-char ABABn  (1 slip)", ABABN_5, True),
    ]:
        margins = []
        for name in names:
            stem = name[:-1]  # e.g. "Lili" or "Moli"
            mention2 = (stem + stem[:2].lower() + "n") if polluted else name
            prompt = f"tenpo pini la jan {name} li lon ma ni. jan {mention2} li pona. tenpo ni la jan {stem}"
            good_id = tok.tok2id[f"c:{name[-1].lower()}"]
            bad_id = tok.tok2id[f"c:{name[0].lower()}"]
            margins.append(logit_margin(model, tok, prompt, good_id, bad_id, device))
        print(
            f"  {label:24s} | mean logit(end)-logit(c:A) = {statistics.mean(margins):+6.2f} "
            f"(sd {statistics.stdev(margins):.2f})"
        )

    # 3. Free-generation copy accuracy & stutter rate (8 samples x 12 names = 96 per group).
    torch.manual_seed(42)
    print("  --- Sampling (96 draws per group, T=0.8) ---")
    for label, names in [
        ("4-char UNIQUE", UNIQUE_4),
        ("4-char ABAB  ", ABAB_4),
        ("5-char UNIQUE", UNIQUE_5),
        ("5-char ABABn ", ABABN_5),
    ]:
        exact = 0
        stutter = 0
        total = 0
        examples = []
        for name in names:
            prompt = f"tenpo pini la jan {name} li lon ma ni. jan {name} li pona. tenpo ni la jan"
            gens = sample_name_copy(model, tok, prompt, device, n_samples=8)
            ab = name[:2].lower()
            for g in gens:
                total += 1
                if g == name:
                    exact += 1
                if g.lower().startswith(ab * 3):
                    stutter += 1
                    if len(examples) < 5:
                        examples.append(f"{name}->{g}")
        ex_str = f" e.g. {', '.join(examples)}" if examples else ""
        print(
            f"  {label} | exact={exact:2d}/{total} ({100*exact/total:4.1f}%) | "
            f"stutter(3+ AB)={stutter:2d}/{total} ({100*stutter/total:4.1f}%){ex_str}"
        )


def main() -> None:
    device = pick_device(None)
    tok = TokiPonaTokenizer()
    for ckpt in [
        Path("checkpoints/smoke"),
        Path("checkpoints/14m"),
        Path("checkpoints/ablation-mix"),
    ]:
        if ckpt.exists():
            eval_checkpoint(ckpt, tok, device)


if __name__ == "__main__":
    main()
