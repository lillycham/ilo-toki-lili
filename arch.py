"""Looped and Depth-Attention (AttnRes) wrapper around LlamaDecoderLayer.

Reuses transformers' LlamaDecoderLayer, LlamaRotaryEmbedding, and LlamaRMSNorm
verbatim so that the 3x2 ablation (unlooped-4L / looped-4Lx3 / unlooped-12L x
standard residual / depth-attention) isolates weight-sharing and depth routing
with zero implementation confounds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


@dataclass
class TokiLlamaConfig:
    vocab_size: int = 192
    hidden_size: int = 384
    intermediate_size: int = 1024
    num_unique_layers: int = 4
    num_loops: int = 1
    num_attention_heads: int = 6
    max_position_embeddings: int = 256
    use_depth_attn: bool = False
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    @property
    def effective_layers(self) -> int:
        return self.num_unique_layers * self.num_loops

    def to_llama_config(self) -> LlamaConfig:
        cfg = LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_unique_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            attention_dropout=self.attention_dropout,
            tie_word_embeddings=self.tie_word_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        cfg._attn_implementation = "sdpa"
        return cfg

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "TokiLlamaConfig":
        return cls(**json.loads(path.read_text()))


class DepthAttention(nn.Module):
    """Block-level Attention Residuals (AttnRes) over previous layer deltas.

    Given sources V = [h_0, delta_1, ..., delta_t] of shape (S, B, T, D):
        K = RMSNorm(V)
        alpha = softmax(<w, K>, dim=0)
        out = sum_i alpha_i * V_i

    Query w is zero-initialized so step 0 starts as a clean uniform average.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = LlamaRMSNorm(hidden_size, eps=eps)
        self.query = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        if len(sources) == 1:
            return sources[0]
        # V: (S, B, T, D)
        v = torch.stack(sources, dim=0)
        k = self.norm(v)
        # Compute depth logits in fp32 for numerical stability under bf16 autocast.
        scores = torch.einsum("d,sbtd->sbt", self.query.float(), k.float())
        weights = F.softmax(scores, dim=0).to(v.dtype)
        return torch.einsum("sbt,sbtd->btd", weights, v)


class TokiLlama(nn.Module):
    def __init__(self, cfg: TokiLlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        llama_cfg = cfg.to_llama_config()

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.rotary_emb = LlamaRotaryEmbedding(config=llama_cfg)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(llama_cfg, layer_idx=i) for i in range(cfg.num_unique_layers)]
        )
        self.norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        if cfg.use_depth_attn:
            # One depth-attn module before each effective layer (except layer 0
            # which only sees h_0) plus one final aggregator before lm_head.
            self.depth_attns = nn.ModuleList(
                [DepthAttention(cfg.hidden_size, eps=cfg.rms_norm_eps) for _ in range(cfg.effective_layers)]
            )
        else:
            self.depth_attns = None

        self.apply(self._init_weights)
        # Re-zero depth-attn queries after generic init.
        if self.depth_attns is not None:
            for da in self.depth_attns:
                nn.init.zeros_(da.query)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        num_loops: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        h = self.embed_tokens(input_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.rotary_emb(h, position_ids)

        loops = num_loops if num_loops is not None else self.cfg.num_loops
        total_steps = self.cfg.num_unique_layers * loops

        if not self.cfg.use_depth_attn:
            for step in range(total_steps):
                layer = self.layers[step % self.cfg.num_unique_layers]
                h = layer(
                    h,
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                if isinstance(h, tuple):
                    h = h[0]
        else:
            assert self.depth_attns is not None
            # sources = [h_0, delta_1, delta_2, ...]
            sources: list[torch.Tensor] = [h]
            for step in range(total_steps):
                h_in = sources[0] if step == 0 else self.depth_attns[step - 1](sources)
                layer = self.layers[step % self.cfg.num_unique_layers]
                h_out = layer(
                    h_in,
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                if isinstance(h_out, tuple):
                    h_out = h_out[0]
                sources.append(h_out - h_in)
            h = self.depth_attns[-1](sources)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 48,
        temperature: float = 0.8,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        out = input_ids
        max_ctx = self.cfg.max_position_embeddings
        for _ in range(max_new_tokens):
            ctx = out[:, -max_ctx:]
            logits, _ = self(ctx)
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs - sorted_probs > top_p
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(sorted_probs, num_samples=1)
                next_id = sorted_idx.gather(-1, sampled)
            out = torch.cat([out, next_id], dim=-1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return out

    def save_pretrained(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.save(out_dir / "toki_config.json")
        torch.save(self.state_dict(), out_dir / "model.pt")

    @classmethod
    def from_pretrained(cls, out_dir: Path, map_location: str | torch.device = "cpu") -> "TokiLlama":
        cfg = TokiLlamaConfig.load(out_dir / "toki_config.json")
        model = cls(cfg)
        state = torch.load(out_dir / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model
