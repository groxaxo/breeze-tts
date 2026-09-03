"""Hybrid bitsandbytes NF4 quantization for Breeze TTS 2.

Quantizes ``nn.Linear`` layers in the text encoder and Qwen3 backbone.
Leaves the depth decoder, codec/vocoder, embeddings, norms, and LM head in
bfloat16 so waveform quality stays close to the official eager path.

CUDA-graph fast path is incompatible with bitsandbytes ``Linear4bit``.
Keep eager execution when NF4 is enabled.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_SKIP_PREFIXES = (
    "depth_decoder",
    "codec_model",
    "embed_text_tokens",
    "lm_head",
)

DEFAULT_SKIP_SUBSTRINGS = (
    "embed_tokens",
    "embed_audio",
    "rotary",
    "rmsnorm",
    "layernorm",
    "layer_norm",
)


@dataclass(frozen=True)
class Nf4Stats:
    replaced: int
    skipped: int
    in_features_min: int
    quant_type: str
    compute_dtype: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "replaced": self.replaced,
            "skipped": self.skipped,
            "in_features_min": self.in_features_min,
            "quant_type": self.quant_type,
            "compute_dtype": self.compute_dtype,
        }


def _is_skipped(
    name: str,
    skip_prefixes: Iterable[str],
    skip_substrings: Iterable[str],
) -> bool:
    lowered = name.lower()
    for prefix in skip_prefixes:
        if name == prefix or name.startswith(f"{prefix}."):
            return True
    return any(token in lowered for token in skip_substrings)


def apply_nf4(
    model: nn.Module,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    skip_prefixes: Iterable[str] = DEFAULT_SKIP_PREFIXES,
    skip_substrings: Iterable[str] = DEFAULT_SKIP_SUBSTRINGS,
    min_features: int = 64,
    include_depth_decoder: bool = False,
) -> Nf4Stats:
    """Replace eligible ``nn.Linear`` modules with bitsandbytes NF4 layers.

    Conversion happens on CPU. Call ``model.to(device)`` afterwards so
    ``Params4bit`` materializes the quantized weights on the GPU.
    """
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit, Params4bit
    except ImportError as exc:
        raise ImportError(
            "NF4 deployment requires bitsandbytes. Install with: "
            "pip install 'bitsandbytes>=0.45.0'"
        ) from exc

    prefixes = tuple(skip_prefixes)
    if include_depth_decoder:
        prefixes = tuple(p for p in prefixes if p != "depth_decoder")
    substrings = tuple(skip_substrings)

    replaced = 0
    skipped = 0

    def recurse(parent: nn.Module, prefix: str) -> None:
        nonlocal replaced, skipped
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if _is_skipped(full_name, prefixes, substrings):
                skipped += sum(
                    1 for module in child.modules() if type(module) is nn.Linear
                )
                continue
            if type(child) is nn.Linear:
                if (
                    child.in_features < min_features
                    or child.out_features < min_features
                ):
                    skipped += 1
                    continue
                quantized = Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                    compress_statistics=True,
                )
                weight = child.weight.detach().to("cpu", dtype=torch.float32)
                quantized.weight = Params4bit(
                    weight,
                    requires_grad=False,
                    quant_type=quant_type,
                    compress_statistics=True,
                )
                if child.bias is not None:
                    quantized.bias = nn.Parameter(
                        child.bias.detach().to("cpu", dtype=compute_dtype),
                        requires_grad=False,
                    )
                parent._modules[child_name] = quantized
                replaced += 1
                continue
            recurse(child, full_name)

    recurse(model, "")
    stats = Nf4Stats(
        replaced=replaced,
        skipped=skipped,
        in_features_min=min_features,
        quant_type=quant_type,
        compute_dtype=str(compute_dtype).replace("torch.", ""),
    )
    logger.info(
        "NF4 applied: replaced=%s skipped=%s quant_type=%s compute_dtype=%s",
        stats.replaced,
        stats.skipped,
        stats.quant_type,
        stats.compute_dtype,
    )
    _ = bnb
    return stats
