#!/usr/bin/env python3
"""APEX quantization recipes for stock llama-quantize.

Python port of localai-org/apex-quant (MIT, see README.md), synced 2026-10-15
against upstream main. Emits a --tensor-type-file, drives an unpatched
llama-quantize, and can predict the resulting GGUF size from the tensor
inventory alone. The bits move by allocation, not by backend changes.

    python tools/quantize/apex/apex.py model-f16.gguf model-apex-balanced.gguf
    python tools/quantize/apex/apex.py --generate-config --profile mini > apex.txt
    python tools/quantize/apex/apex.py --estimate model-f16.gguf --profile mini
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PROFILE = "balanced"
DEFAULT_LAYERS = 40
DEFAULT_BASE_TYPE = "Q6_K"

# Layer bands, from the APEX reference scripts. Edges are the first/last few
# layers, which do embedding alignment and logit generation and are the most
# sensitive; the middle is the most redundant and takes the hardest cut.
EDGE_HI = 4
NEAR_HI = 9
ATTN_EDGE_HI = 2
ATTN_EDGE_TAIL = 3

# profile -> routed-expert types by band (edge/near/mid) plus shared and
# attention bands. Blanks take the Q6_K/Q5_K/Q8_0 defaults below.
MOE_PROFILES = {
    "quality":    dict(mid_exp="IQ4_XS", base="Q6_K"),
    "i-quality":  dict(mid_exp="IQ4_XS", base="Q6_K"),
    "balanced":   dict(mid_exp="Q5_K",   base="Q6_K"),
    "i-balanced": dict(mid_exp="Q5_K",   base="Q6_K"),
    "compact":    dict(edge_exp="Q4_K", near_exp="Q3_K", mid_exp="Q3_K",
                       edge_shared="Q6_K", mid_shared="Q6_K",
                       edge_attn="Q4_K", mid_attn="Q4_K", base="Q4_K_M"),
    "i-compact":  dict(edge_exp="Q4_K", near_exp="Q3_K", mid_exp="Q3_K",
                       edge_shared="Q6_K", mid_shared="Q6_K",
                       edge_attn="Q4_K", mid_attn="Q4_K", base="Q4_K_M"),
    "mini":       dict(edge_exp="Q3_K", near_exp="Q3_K", mid_exp="IQ2_S",
                       edge_shared="Q5_K", mid_shared="Q4_K",
                       edge_attn="Q4_K", mid_attn="Q3_K", base="Q3_K_M"),
    "nano":       dict(edge_exp="Q3_K", near_exp="IQ2_S", mid_exp="IQ2_XXS",
                       edge_shared="Q5_K", mid_shared="Q4_K",
                       edge_attn="Q4_K", mid_attn="Q3_K", base="Q3_K_M"),
    "i-nano":     dict(edge_exp="Q3_K", near_exp="IQ2_S", mid_exp="IQ2_XXS",
                       edge_shared="Q5_K", mid_shared="Q4_K",
                       edge_attn="Q4_K", mid_attn="Q3_K", base="Q3_K_M"),
    "micro":      dict(edge_exp="Q3_K", near_exp="IQ2_XS", mid_exp="IQ1_M",
                       edge_shared="Q5_K", mid_shared="Q4_K",
                       edge_attn="Q4_K", mid_attn="Q3_K", base="Q3_K_M"),
    "i-micro":    dict(edge_exp="Q3_K", near_exp="IQ2_XS", mid_exp="IQ1_M",
                       edge_shared="Q5_K", mid_shared="Q4_K",
                       edge_attn="Q4_K", mid_attn="Q3_K", base="Q3_K_M"),
}

# Dense models carry no expert tensors, so the levers are the FFN band and the
# attention split. Values are (edge, near, mid).
DENSE_PROFILES = {
    "dense-flat": dict(
        gate=("IQ4_XS", "IQ4_XS", "IQ4_XS"), up=("Q5_K", "Q5_K", "Q5_K"),
        down=("Q5_K", "Q5_K", "Q5_K"), base="Q4_K_M"),
    "dense-grad": dict(
        gate=("Q5_K", "Q4_K", "IQ4_XS"), up=("Q6_K", "Q5_K", "Q4_K"),
        down=("Q6_K", "Q5_K", "Q5_K"), base="Q4_K_M"),
    "dense-hybrid": dict(
        gate=("Q5_K", "Q4_K", "IQ4_XS"), up=("Q6_K", "Q5_K", "Q4_K"),
        down=("Q6_K", "Q5_K", "Q5_K"), base="Q4_K_M"),
    "dense-hybrid-quality": dict(
        gate=("Q6_K", "Q5_K", "Q5_K"), up=("Q6_K", "Q6_K", "Q5_K"),
        down=("Q8_0", "Q6_K", "Q6_K"), base="Q5_K_M"),
}

DENSE_ATTN_DEFAULTS = dict(
    linattn="Q5_K", linattn_small="Q4_K", fullattn="Q5_K", fullattn_v="Q6_K",
    embd="Q4_K", output="Q6_K")

# Allocations built from the measured per-group sensitivity curve (dKL per GB
# saved) in upstream's generate_opt_config.py. These are size-matched arms
# from the Qwen3.8-27B experiment, not new claims.
OPT_PROFILES = {
    "opt-balanced": dict(
        token_embd="Q3_K", output="Q8_0", fullattn="Q6_K",
        linattn_out="Q6_K", linattn_qkv="Q5_K", linattn_gate="Q4_K",
        ssm_small="Q4_K", ffn_edge=("Q6_K", "Q6_K", "Q6_K"),
        ffn_near=("Q5_K", "Q5_K", "Q5_K"),
        ffn_mid=("IQ4_XS", "Q4_K", "Q4_K")),
    "opt-mini": dict(
        token_embd="IQ2_S", output="Q6_K", fullattn="Q5_K",
        linattn_out="Q5_K", linattn_qkv="Q4_K", linattn_gate="Q3_K",
        ssm_small="Q3_K", ffn_edge=("Q4_K", "Q4_K", "Q5_K"),
        ffn_near=("Q3_K", "Q3_K", "Q4_K"),
        ffn_mid=("IQ3_XXS", "Q3_K", "Q3_K")),
    "opt-mini-v2": dict(
        token_embd="IQ2_S", output="Q8_0", fullattn="Q5_K",
        linattn_out="Q5_K", linattn_qkv="Q4_K", linattn_gate="Q3_K",
        ssm_small="Q3_K", ffn_edge=("Q4_K", "Q4_K", "Q5_K"),
        ffn_near=("Q3_K", "Q3_K", "Q4_K"),
        ffn_mid=("IQ3_XXS", "IQ3_XXS", "IQ3_XXS")),
    "opt-compact": dict(
        token_embd="IQ2_S", output="Q6_K", fullattn="Q5_K",
        linattn_out="Q5_K", linattn_qkv="Q5_K", linattn_gate="Q4_K",
        ssm_small="Q4_K", ffn_edge=("Q5_K", "Q5_K", "Q5_K"),
        ffn_near=("Q4_K", "Q4_K", "Q4_K"),
        ffn_mid=("Q3_K", "Q3_K", "Q4_K")),
    "opt-compact-flat": dict(
        token_embd="Q3_K", output="Q6_K", fullattn="Q5_K",
        linattn_out="Q5_K", linattn_qkv="Q5_K", linattn_gate="Q5_K",
        ssm_small="Q5_K", ffn_edge=("Q3_K", "Q4_K", "Q4_K"),
        ffn_near=("Q3_K", "Q4_K", "Q4_K"),
        ffn_mid=("Q3_K", "Q4_K", "Q4_K")),
    "opt-nano": dict(
        token_embd="IQ2_S", output="Q6_K", fullattn="Q4_K",
        linattn_out="Q4_K", linattn_qkv="Q3_K", linattn_gate="IQ3_XXS",
        ssm_small="IQ3_XXS", ffn_edge=("Q3_K", "Q3_K", "Q4_K"),
        ffn_near=("IQ3_XXS", "IQ3_XXS", "Q3_K"),
        ffn_mid=("IQ2_XXS", "IQ2_S", "IQ2_S")),
    "opt-nano-flat": dict(
        token_embd="IQ2_S", output="Q6_K", fullattn="Q4_K",
        linattn_out="Q4_K", linattn_qkv="Q4_K", linattn_gate="Q4_K",
        ssm_small="Q4_K", ffn_edge=("IQ2_S", "IQ2_S", "IQ2_S"),
        ffn_near=("IQ2_S", "IQ2_S", "IQ2_S"),
        ffn_mid=("IQ2_S", "IQ2_S", "IQ2_S")),
    "opt-mini-flat": dict(
        token_embd="Q3_K", output="Q6_K", fullattn="Q5_K",
        linattn_out="Q5_K", linattn_qkv="Q5_K", linattn_gate="Q5_K",
        ssm_small="Q5_K", ffn_edge=("IQ3_XXS", "Q3_K", "Q3_K"),
        ffn_near=("IQ3_XXS", "Q3_K", "Q3_K"),
        ffn_mid=("IQ3_XXS", "Q3_K", "Q3_K")),
}

# Tiers that gain their accuracy from an importance matrix.
NEEDS_IMATRIX = {
    "i-quality", "i-balanced", "i-compact", "i-nano", "i-micro",
    "mini", "nano", "micro",
}

PROFILE_LIST = (
    "MoE:    quality, i-quality, balanced, i-balanced, compact, i-compact,\n"
    "        mini, nano, i-nano, micro, i-micro, custom\n"
    "dense:  dense-flat, dense-grad, dense-hybrid, dense-hybrid-quality\n"
    "opt:    opt-balanced, opt-mini, opt-mini-v2, opt-compact,\n"
    "        opt-compact-flat, opt-nano, opt-nano-flat, opt-mini-flat"
)

_MOE_DEFAULTS = dict(
    edge_exp="Q6_K", near_exp="Q5_K", edge_shared="Q8_0", mid_shared="Q8_0",
    edge_attn="Q6_K", mid_attn="Q6_K")

# Effective bits per weight including super-block overhead, from upstream's
# estimate_config_size.py (validated against three published Qwen3.8-27B GGUFs
# to within 0.1%; Garuda has not re-verified that claim).
BPW = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q4_K": 4.5,
    "Q5_0": 5.5, "Q4_0": 4.5,
    "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "Q3_K": 3.4375, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625,
    "Q2_K": 2.625, "IQ2_S": 2.5, "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625,
    "IQ1_M": 1.75, "IQ1_S": 1.5625,
}

# Norms and biases below this many elements are kept at F32 by llama.cpp.
F32_FLOOR = 100_000

OPT_BASE_TYPE = "Q5_K"

GGUF_MAGIC = b"GGUF"
_GGUF_SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8,
                     11: 8, 12: 8}

WANTED_KEYS = ("block_count", "expert_used_count", "expert_count",
               "general.architecture")


def warn(msg: str) -> None:
    print("WARNING: %s" % msg, file=sys.stderr)


def info(msg: str) -> None:
    print(">>> %s" % msg)


def die(msg: str, code: int = 1) -> int:
    print("ERROR: %s" % msg, file=sys.stderr)
    return code


def resolve_moe_profile(name: str, args: argparse.Namespace) -> dict:
    if name == "custom":
        # One explicit type per role; the bands mirror the edge choice.
        edge_exp = args.edge_exp
        edge_shared = args.edge_shared or "Q8_0"
        edge_attn = args.edge_attn or "Q6_K"
        return dict(edge_exp=edge_exp,
                    near_exp=args.near_exp or edge_exp,
                    mid_exp=args.mid_exp or edge_exp,
                    edge_shared=edge_shared,
                    mid_shared=args.mid_shared or edge_shared,
                    edge_attn=edge_attn,
                    mid_attn=args.mid_attn or edge_attn,
                    base=args.base_type or DEFAULT_BASE_TYPE)
    if name not in MOE_PROFILES:
        return {}
    spec = dict(_MOE_DEFAULTS)
    spec.update(MOE_PROFILES[name])
    spec.update(near_exp=args.near_exp or spec["near_exp"],
                mid_exp=args.mid_exp or spec["mid_exp"],
                edge_shared=args.edge_shared or spec["edge_shared"],
                mid_shared=args.mid_shared or spec["mid_shared"],
                edge_attn=args.edge_attn or spec["edge_attn"],
                mid_attn=args.mid_attn or spec["mid_attn"])
    if args.base_type:
        spec["base"] = args.base_type
    return spec


def resolve_dense_profile(name: str, args: argparse.Namespace) -> dict:
    spec = dict(DENSE_ATTN_DEFAULTS)
    spec.update(DENSE_PROFILES[name])
    if name.startswith("dense-hybrid"):
        # The structural lever: full-attention layers are ~6% of parameters and
        # linear-attention ~20%, so pinning the former pays for cutting the
        # latter. Applies to dense-hybrid and dense-hybrid-quality.
        spec.update(fullattn="Q8_0", fullattn_v="Q8_0", linattn="Q4_K",
                    linattn_small="Q4_K")
    if args.linattn:
        spec["linattn"] = args.linattn
    if args.fullattn:
        spec["fullattn"] = args.fullattn
    if args.embd:
        spec["embd"] = args.embd
    if args.output_type:
        spec["output"] = args.output_type
    if args.base_type:
        spec["base"] = args.base_type
    return spec


def resolve_opt_profile(name: str, args: argparse.Namespace) -> dict:
    spec = dict(OPT_PROFILES[name])
    spec["base"] = args.base_type or OPT_BASE_TYPE
    if getattr(args, "fullattn", None):
        spec["fullattn"] = args.fullattn
    if getattr(args, "linattn", None):
        spec["linattn_qkv"] = args.linattn
    if getattr(args, "embd", None):
        spec["token_embd"] = args.embd
    if getattr(args, "output_type", None):
        spec["output"] = args.output_type
    return spec


def emit_moe(spec: dict, layers: int, dense_layers: int) -> list[str]:
    out = []
    for i in range(layers):
        edge = i <= EDGE_HI or i >= layers - 5
        near = i <= NEAR_HI or i >= layers - 10
        exp_type = spec["edge_exp"] if edge else (
            spec["near_exp"] if near else spec["mid_exp"])
        shared_type = spec["edge_shared"] if edge else spec["mid_shared"]
        attn_edge = i <= ATTN_EDGE_HI or i >= layers - ATTN_EDGE_TAIL
        attn_type = spec["edge_attn"] if attn_edge else spec["mid_attn"]

        if i < dense_layers:
            # Leading dense (non-MoE) FFN layers keep shared-expert precision.
            # The .weight suffix stops the rule matching ffn_gate_inp.
            out.append("blk.%d.ffn_gate.weight=%s" % (i, shared_type))
            out.append("blk.%d.ffn_up.weight=%s" % (i, shared_type))
            out.append("blk.%d.ffn_down.weight=%s" % (i, shared_type))
        else:
            out.append("blk.%d.ffn_gate_exps=%s" % (i, exp_type))
            out.append("blk.%d.ffn_up_exps=%s" % (i, exp_type))
            out.append("blk.%d.ffn_down_exps=%s" % (i, exp_type))

        out.append("blk.%d.ffn_gate_shexp=%s" % (i, shared_type))
        out.append("blk.%d.ffn_up_shexp=%s" % (i, shared_type))
        out.append("blk.%d.ffn_down_shexp=%s" % (i, shared_type))

        out.append("blk.%d.attn_q=%s" % (i, attn_type))
        out.append("blk.%d.attn_k=%s" % (i, attn_type))
        out.append("blk.%d.attn_v=%s" % (i, attn_type))
        out.append("blk.%d.attn_output=%s" % (i, attn_type))
        out.append("blk.%d.attn_gate=%s" % (i, attn_type))
        out.append("blk.%d.attn_qkv=%s" % (i, attn_type))

        out.append("blk.%d.shortconv.in_proj=%s" % (i, attn_type))
        out.append("blk.%d.shortconv.out_proj=%s" % (i, attn_type))

        out.append("blk.%d.ssm_alpha=%s" % (i, attn_type))
        out.append("blk.%d.ssm_beta=%s" % (i, attn_type))
        out.append("blk.%d.ssm_out=%s" % (i, attn_type))
    return out


def _esc(name: str) -> str:
    return name.replace(".", "\\.")


def emit_dense(spec: dict, layers: int) -> list[str]:
    out = []
    # Anchored: llama-quantize regex-searches each tensor name unanchored, so
    # an unanchored "output.weight" would also capture every attn_output.weight
    # and silently mis-quantize the attention output projections.
    out.append("^token_embd\\.weight$=%s" % spec["embd"])
    out.append("^output\\.weight$=%s" % spec["output"])
    for i in range(layers):
        edge = i <= EDGE_HI or i >= layers - 5
        near = i <= NEAR_HI or i >= layers - 10
        band = 0 if edge else (1 if near else 2)
        out.append("^blk\\.%d\\.ffn_gate\\.weight$=%s" % (i, spec["gate"][band]))
        out.append("^blk\\.%d\\.ffn_up\\.weight$=%s" % (i, spec["up"][band]))
        out.append("^blk\\.%d\\.ffn_down\\.weight$=%s" % (i, spec["down"][band]))

        out.append("^blk\\.%d\\.attn_q\\.weight$=%s" % (i, spec["fullattn"]))
        out.append("^blk\\.%d\\.attn_k\\.weight$=%s" % (i, spec["fullattn"]))
        out.append("^blk\\.%d\\.attn_v\\.weight$=%s" % (i, spec["fullattn_v"]))
        out.append("^blk\\.%d\\.attn_output\\.weight$=%s" % (i, spec["fullattn"]))

        out.append("^blk\\.%d\\.attn_qkv\\.weight$=%s" % (i, spec["linattn"]))
        out.append("^blk\\.%d\\.attn_gate\\.weight$=%s" % (i, spec["linattn"]))
        out.append("^blk\\.%d\\.ssm_out\\.weight$=%s" % (i, spec["linattn"]))
        out.append("^blk\\.%d\\.ssm_alpha\\.weight$=%s"
                   % (i, spec["linattn_small"]))
        out.append("^blk\\.%d\\.ssm_beta\\.weight$=%s"
                   % (i, spec["linattn_small"]))
    return out


def emit_opt(spec: dict, layers: int) -> list[str]:
    out = ["^token_embd\\.weight$=%s" % spec["token_embd"],
           "^output\\.weight$=%s" % spec["output"]]
    for i in range(layers):
        edge = i <= EDGE_HI or i >= layers - 5
        near = i <= NEAR_HI or i >= layers - 10
        band = 0 if edge else (1 if near else 2)
        for name, qtype in zip(("ffn_gate", "ffn_up", "ffn_down"),
                               spec["ffn_%s" % ("edge" if edge else
                                                "near" if near else "mid")]):
            out.append("^blk\\.%d\\.%s\\.weight$=%s" % (i, name, qtype))
        for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
            out.append("^blk\\.%d\\.%s\\.weight$=%s"
                       % (i, name, spec["fullattn"]))
        out.append("^blk\\.%d\\.attn_qkv\\.weight$=%s"
                   % (i, spec["linattn_qkv"]))
        out.append("^blk\\.%d\\.attn_gate\\.weight$=%s"
                   % (i, spec["linattn_gate"]))
        out.append("^blk\\.%d\\.ssm_out\\.weight$=%s"
                   % (i, spec["linattn_out"]))
        out.append("^blk\\.%d\\.ssm_alpha\\.weight$=%s"
                   % (i, spec["ssm_small"]))
        out.append("^blk\\.%d\\.ssm_beta\\.weight$=%s"
                   % (i, spec["ssm_small"]))
    return out


def profile_kind(profile: str, args: argparse.Namespace) -> str:
    """moe | dense | opt. Dies on an architecture/profile mismatch."""
    if profile in OPT_PROFILES:
        return "opt"
    if profile in DENSE_PROFILES:
        return "dense"
    if args.arch == "dense":
        die("--arch dense needs a dense or opt profile; got '%s'" % profile)
        raise SystemExit(1)
    return "moe"


def emit_config(profile: str, args: argparse.Namespace,
                layers: int) -> list[str]:
    kind = profile_kind(profile, args)
    if kind == "opt":
        return emit_opt(resolve_opt_profile(profile, args), layers)
    if kind == "dense":
        return emit_dense(resolve_dense_profile(profile, args), layers)
    spec = resolve_moe_profile(profile, args)
    if not spec:
        return []
    return emit_moe(spec, layers, args.dense_layers)


def base_type_for(profile: str, args: argparse.Namespace) -> str:
    kind = profile_kind(profile, args)
    if kind == "opt":
        return resolve_opt_profile(profile, args)["base"]
    if kind == "dense":
        return resolve_dense_profile(profile, args)["base"]
    return resolve_moe_profile(profile, args)["base"]


# ------------------------------------------------------------------ gguf input

def _gguf_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _gguf_skip(f, vtype: int) -> None:
    size = _GGUF_SCALAR_SIZE.get(vtype)
    if size is not None:
        f.seek(size, os.SEEK_CUR)
        return
    if vtype == 8:
        _gguf_string(f)
        return
    if vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        esize = _GGUF_SCALAR_SIZE.get(etype)
        if esize is not None:
            f.seek(esize * count, os.SEEK_CUR)
        else:
            for _ in range(count):
                _gguf_skip(f, etype)
        return
    raise ValueError("unknown gguf value type %d" % vtype)


def read_gguf_meta(path: Path, keys=WANTED_KEYS) -> dict | None:
    """Read a few metadata keys without parsing tensor info. None if unreadable.

    Stops as soon as every wanted key is seen, because the tokenizer arrays at
    the end of a real file are large. Layer keys are matched by suffix, since
    llama.cpp writes them as "<arch>.block_count".
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return None
            (version,) = struct.unpack("<I", f.read(4))
            if version == 1:
                _, n_kv = struct.unpack("<II", f.read(8))
            else:
                _, n_kv = struct.unpack("<QQ", f.read(16))
            meta = {}
            for _ in range(n_kv):
                key = _gguf_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                name = None
                for k in keys:
                    if key == k or key.endswith("." + k):
                        name = k
                        break
                if name is not None:
                    if vtype == 4:
                        meta[name] = struct.unpack("<I", f.read(4))[0]
                    elif vtype == 5:
                        meta[name] = struct.unpack("<i", f.read(4))[0]
                    elif vtype == 10:
                        meta[name] = struct.unpack("<Q", f.read(8))[0]
                    elif vtype == 11:
                        meta[name] = struct.unpack("<q", f.read(8))[0]
                    elif vtype == 8:
                        meta[name] = _gguf_string(f)
                    else:
                        _gguf_skip(f, vtype)
                else:
                    _gguf_skip(f, vtype)
                if all(k in meta for k in keys):
                    break
            return meta
    except (OSError, ValueError, struct.error):
        return None


def read_gguf_tensors(path: Path) -> list[tuple[str, int, int]] | None:
    """(name, numel, n_dims) for every tensor, from the tensor-info section.

    The tensor info sits after the metadata section, so the KVs are skipped
    with the same type-aware walk read_gguf_meta uses.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return None
            (version,) = struct.unpack("<I", f.read(4))
            if version == 1:
                n_tensors, n_kv = struct.unpack("<II", f.read(8))
            else:
                n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            for _ in range(n_kv):
                _gguf_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                _gguf_skip(f, vtype)
            out = []
            for _ in range(n_tensors):
                name = _gguf_string(f)
                (n_dims,) = struct.unpack("<I", f.read(4))
                dims = struct.unpack("<%dQ" % n_dims,
                                     f.read(8 * n_dims)) if n_dims else ()
                f.read(4)   # ggml type
                f.read(8)   # offset
                numel = 1
                for d in dims:
                    numel *= d
                out.append((name, numel, n_dims))
            return out
    except (OSError, ValueError, struct.error):
        return None


def allows_quantization(name: str, n_dims: int) -> bool:
    """llama.cpp's own gate, src/llama-quant.cpp:287."""
    if n_dims < 2:
        return False
    if not name.endswith("weight"):
        return False
    return "_norm.weight" not in name


# ------------------------------------------------------------------- estimating

def match_type(name: str, rules: list[tuple[str, str]]) -> str | None:
    """llama-quantize semantics: entries are regex-matched, first match wins."""
    for pattern, qtype in rules:
        try:
            if re.search(pattern, name):
                return qtype
        except re.error:
            if pattern == name:
                return qtype
    return None


def load_rules(path: Path) -> list[tuple[str, str]]:
    rules = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        pattern, qtype = line.rsplit("=", 1)
        rules.append((pattern.strip(), qtype.strip().upper()))
    return rules


def estimate_size(tensors: list[tuple[str, int, int]],
                  rules: list[tuple[str, str]], base: str) -> tuple:
    """(GB, {type: params}, uncovered_count). Raises KeyError on a bad type."""
    total_bits = 0.0
    by_type: dict[str, int] = {}
    uncovered = 0
    for name, numel, n_dims in tensors:
        if not allows_quantization(name, n_dims) or numel < F32_FLOOR:
            total_bits += numel * BPW["F32"]
            by_type["F32 (kept)"] = by_type.get("F32 (kept)", 0) + numel
            continue
        qtype = match_type(name, rules)
        if qtype is None:
            qtype = base
            uncovered += 1
        if qtype not in BPW:
            raise KeyError(qtype)
        total_bits += numel * BPW[qtype]
        by_type[qtype] = by_type.get(qtype, 0) + numel
    return total_bits / 8 / 1e9, by_type, uncovered


# ---------------------------------------------------------------------- driver

def find_quantize(explicit: str | None) -> str | None:
    candidates = [explicit, os.environ.get("LLAMA_QUANTIZE")]
    llm_dir = os.environ.get("LLAMA_CPP_DIR")
    if llm_dir:
        candidates.append(str(Path(llm_dir) / "llama-quantize"))
    for pattern in ("build/*/bin/llama-quantize*", "build/bin/llama-quantize*"):
        for cfg in sorted(REPO_ROOT.glob(pattern)):
            candidates.append(str(cfg))
    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(cand)
    return shutil.which("llama-quantize")


def detect_layers(path: Path, args: argparse.Namespace) -> int:
    """Layer count: --layers, else the GGUF's own block_count, else 40."""
    if args.layers:
        return args.layers
    meta = read_gguf_meta(path)
    if meta and "block_count" in meta:
        return int(meta["block_count"])
    warn("could not read block_count from %s; assuming %d (override with "
         "--layers)" % (path, DEFAULT_LAYERS))
    return DEFAULT_LAYERS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apex.py",
        description="APEX quantization for llama-quantize (MoE-aware mixed "
                    "precision). Default profile: %s." % DEFAULT_PROFILE)
    p.add_argument("input", nargs="?", help="input GGUF (F16 or F32)")
    p.add_argument("output", nargs="?", help="output GGUF (default: "
               "<input>-apex-<profile>.gguf)")
    p.add_argument("-p", "--profile", default=DEFAULT_PROFILE,
                   help="APEX profile (default: %(default)s). " + PROFILE_LIST)
    p.add_argument("-l", "--layers", type=int, default=0,
                   help="transformer layers (default: block_count from the "
                        "input GGUF, else %d)" % DEFAULT_LAYERS)
    p.add_argument("--arch", choices=["moe", "dense"], default="moe",
                   help="model kind (default: %(default)s); dense and opt "
                        "profiles force dense")
    p.add_argument("--dense-layers", type=int, default=0,
                   help="leading dense (non-MoE) FFN layers inside a MoE")
    p.add_argument("-c", "--config", help="use this tensor-type file verbatim")
    p.add_argument("-i", "--imatrix", help="importance matrix file")
    p.add_argument("-b", "--base-type",
                   help="base quant type; default is the profile's own")
    p.add_argument("--generate-config", action="store_true",
                   help="write the tensor-type file and exit")
    p.add_argument("--estimate", action="store_true",
                   help="predict the GGUF size for this config and exit")
    p.add_argument("-o", help="output file for --generate-config")
    p.add_argument("--quantize", help="path to llama-quantize")
    p.add_argument("--threads", type=int, default=0,
                   help="threads for llama-quantize")
    p.add_argument("--leave-output-tensor", action="store_true")
    p.add_argument("--allow-requantize", action="store_true")
    p.add_argument("--keep-split", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="llama-quantize --dry-run: size only, no output")
    p.add_argument("--quiet", action="store_true",
                   help="with --estimate, print only the size in GB")
    p.add_argument("--edge-exp")
    p.add_argument("--near-exp")
    p.add_argument("--mid-exp")
    p.add_argument("--edge-shared")
    p.add_argument("--mid-shared")
    p.add_argument("--edge-attn")
    p.add_argument("--mid-attn")
    p.add_argument("--linattn", help="dense/opt: linear-attention tensor type")
    p.add_argument("--fullattn", help="dense/opt: full-attention tensor type")
    p.add_argument("--embd", help="dense/opt: token embedding type")
    p.add_argument("--output-type", help="dense/opt: output tensor type")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    profile = args.profile

    known = set(MOE_PROFILES) | set(DENSE_PROFILES) | set(OPT_PROFILES) \
        | {"custom"}
    if profile not in known:
        return die("unknown profile '%s'\nAvailable:\n  %s"
                   % (profile, PROFILE_LIST))
    kind = profile_kind(profile, args)
    if kind != "moe" and args.dense_layers:
        return die("--dense-layers counts leading dense FFN layers inside a "
                   "MoE; it is incompatible with the %s profile '%s'"
                   % (kind, profile))
    if args.config and args.generate_config:
        return die("--config cannot be combined with --generate-config")
    if args.dense_layers < 0:
        return die("--dense-layers must be >= 0")
    if profile == "custom" and not args.edge_exp:
        return die("--custom requires --edge-exp")

    rules = None
    if args.config:
        tt_file = Path(args.config)
        if not tt_file.is_file():
            return die("config file not found: %s" % args.config)
        rules = load_rules(tt_file)
        layers = args.layers or DEFAULT_LAYERS
        lines = None
    else:
        if args.generate_config or args.estimate:
            layers = args.layers or DEFAULT_LAYERS
        elif args.input:
            layers = detect_layers(Path(args.input), args)
        else:
            return die("input GGUF required, or use --generate-config")
        lines = emit_config(profile, args, layers)
        if not lines:
            return die("profile '%s' produced no config" % profile)
        rules = []
        for ln in lines:
            name, _, qtype = ln.partition("=")
            rules.append((name, qtype))
        if args.generate_config:
            text = "\n".join(lines) + "\n"
            if args.o:
                Path(args.o).write_text(text, encoding="utf-8")
                info("Config written to: %s (%d lines, %d layers)"
                     % (args.o, len(lines), layers))
            else:
                sys.stdout.write(text)
            return 0
        fd, tmp = tempfile.mkstemp(prefix="apex-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        tt_file = Path(tmp)

    if not args.input:
        return die("input GGUF required")
    if not Path(args.input).is_file():
        return die("input file not found: %s" % args.input)

    if args.estimate:
        tensors = read_gguf_tensors(Path(args.input))
        if tensors is None:
            return die("could not read the tensor inventory from %s"
                       % args.input)
        base = base_type_for(profile, args)
        try:
            size_gb, by_type, uncovered = estimate_size(tensors, rules, base)
        except KeyError as exc:
            return die("type %s is not in the bits-per-weight table; use a "
                       "type the estimator knows" % exc)
        total_params = sum(n for _, n, _ in tensors)
        if args.quiet:
            print("%.3f" % size_gb)
            return 0
        info("=== APEX Estimate ===")
        info("Profile:   %s (base %s)" % (profile, base))
        info("Input:     %s" % args.input)
        info("Inventory: %d tensors, %.2f B params"
             % (len(tensors), total_params / 1e9))
        info("Rules:     %d  (uncovered fall back to %s: %d)"
             % (len(rules), base, uncovered))
        print("")
        print("estimated size: %.2f GB" % size_gb)
        print("")
        print("%-14s%12s%9s" % ("type", "params", "share"))
        for qtype, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
            print("%-14s%9.3f B%7.1f%%"
                  % (qtype, n / 1e9, 100 * n / total_params))
        return 0

    output = args.output
    if not output:
        stem = Path(args.input).stem
        output = str(Path(args.input).parent
                     / ("%s-apex-%s.gguf" % (stem, profile)))

    meta = read_gguf_meta(Path(args.input))
    if profile in NEEDS_IMATRIX and not args.imatrix:
        warn("profile '%s' benefits from --imatrix. Continuing without it."
             % profile)
    if meta is not None:
        has_experts = ("expert_used_count" in meta or "expert_count" in meta)
        if kind == "moe" and not has_experts:
            warn("no expert tensors in %s: MoE profiles assume them. Use a "
                 "dense or opt profile." % args.input)
        if kind != "moe" and has_experts:
            warn("%s has expert tensors: %s profiles assume none, so routed "
                 "experts fall back to base type." % (args.input, kind))
        arch = meta.get("general.architecture", "")
        if arch.startswith("lfm2") and not args.dense_layers:
            warn("arch %s may have leading dense FFN layers; pass "
                 "--dense-layers N if it does." % arch)

    quantize = find_quantize(args.quantize)
    if not quantize:
        return die("llama-quantize not found. Set --quantize or "
                   "LLAMA_QUANTIZE, or build with LLAMA_BUILD_TOOLS=ON.")

    cmd = [quantize, "--tensor-type-file", str(tt_file)]
    if args.imatrix:
        cmd += ["--imatrix", args.imatrix]
    if args.leave_output_tensor:
        cmd.append("--leave-output-tensor")
    if args.allow_requantize:
        cmd.append("--allow-requantize")
    if args.keep_split:
        cmd.append("--keep-split")
    if args.dry_run:
        cmd.append("--dry-run")
    cmd += [args.input, output, base_type_for(profile, args)]
    if args.threads:
        cmd.append(str(args.threads))

    info("=== APEX Quantize ===")
    info("Profile:   %s" % profile)
    info("Base type: %s" % base_type_for(profile, args))
    info("Input:     %s" % args.input)
    info("Output:    %s" % output)
    if args.imatrix:
        info("Imatrix:   %s" % args.imatrix)
    if lines is None:
        info("Config:    %s" % tt_file)
    else:
        info("Config:    %d rules, %d layers" % (len(lines), layers))
    info("")

    try:
        return subprocess.call(cmd)
    finally:
        if lines is not None:
            try:
                os.unlink(tt_file)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
