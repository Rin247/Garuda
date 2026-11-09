# APEX quantization

APEX -- *Adaptive Precision for EXpert models* -- assigns a quantization type per
tensor **role** (routed expert / shared expert / attention) and per **layer
position** (edges sensitive, middle redundant). It is a recipe over stock
`llama-quantize --tensor-type-file`: no patched build, no new kernels.

Upstream: <https://github.com/localai-org/apex-quant> (LocalAI, MIT). Garuda's
port is `apex.py` -- one stdlib-only Python file, synced 2026-10-15 against
upstream `main`. The tensor-name fixtures in `tests/fixtures/` and the
bits-per-weight table are vendored from the same repo.

- `apex.py` -- generate a config, quantize, or predict the output size
- `test_apex.py` -- stdlib unittest; mirrors llama-quantize's matcher exactly
- `tests/fixtures/` -- real tensor-name lists from published GGUF headers

## Usage

```bash
# Generate a config only
python tools/quantize/apex/apex.py --generate-config --profile balanced > apex.txt

# Quantize (finds llama-quantize via --quantize, $LLAMA_QUANTIZE, $LLAMA_CPP_DIR,
# then build/*/bin/, then PATH)
python tools/quantize/apex/apex.py model-f16.gguf model-apex-balanced.gguf
python tools/quantize/apex/apex.py --profile mini --imatrix imatrix.dat model-f16.gguf out.gguf

# Predict the GGUF size without quantizing (reads the tensor inventory from
# the input file itself)
python tools/quantize/apex/apex.py --estimate --profile mini model-f16.gguf
python tools/quantize/apex/apex.py --estimate --profile mini --quiet model-f16.gguf

# Bring your own config
python tools/quantize/apex/apex.py --config my-config.txt model-f16.gguf out.gguf

# Tests
python tools/quantize/apex/test_apex.py
```

The layer count is read from the GGUF's own `block_count` (any arch-prefixed
key), so no `NUM_LAYERS` environment variable and no wrong default. Override
with `--layers N`.

## Tiers

MoE (routed experts cut hardest, shared expert kept high):

| profile | expert bands | shared | attention | base |
|---|---|---|---|---|
| quality | Q6_K / Q5_K / IQ4_XS | Q8_0 | Q6_K | Q6_K |
| balanced | Q6_K / Q5_K / Q5_K | Q8_0 | Q6_K | Q6_K |
| compact | Q4_K / Q3_K / Q3_K | Q6_K | Q4_K | Q4_K_M |
| mini | Q3_K / Q3_K / IQ2_S | Q5_K / Q4_K | Q4_K / Q3_K | Q3_K_M |
| nano | Q3_K / IQ2_S / IQ2_XXS | Q5_K / Q4_K | Q4_K / Q3_K | Q3_K_M |
| micro | Q3_K / IQ2_XS / IQ1_M | Q5_K / Q4_K | Q4_K / Q3_K | Q3_K_M |
| custom | `--edge-exp` etc. | ... | ... | Q6_K |

`i-` variants are the same allocation with a diverse-calibration importance
matrix (`--imatrix`); `mini`, `nano` and `micro` also benefit from one. Bands
are edge = first/last 5 layers, near = next 5, mid = the rest; attention edges
are the first/last 3.

Dense and hybrid-attention models (no routed experts; the levers are the FFN
band and the attention split). These are the size-matched arms of upstream's
Qwen3.8-27B experiment:

| profile | shape |
|---|---|
| dense-flat | flat per-role FFN, no layer gradient -- the shelf-quant control |
| dense-grad | + FFN layer gradient, attention identical to the control |
| dense-hybrid | + full-attention pinned to Q8_0, linear-attention at Q4_K |
| dense-hybrid-quality | the same attention split in the Q5/Q6 band |
| opt-* | allocations fitted to the measured dKL-per-GB sensitivity curve |

Upstream's own measurement on Qwen3.8-27B is the caveat that matters: the
measured allocation **loses** to a flat one above roughly 15.2 GB (balanced
17.65 GB, KL 0.0112 vs flat 0.0098) and only wins below it (mini 13.49 GB,
0.0490 vs 0.0654). Those are upstream's numbers, not Garuda's; re-run the sweep
before trusting them on another model.

## Differences from upstream

Kept faithful: every profile table, the band boundaries, the emitter output,
the `--config` passthrough, the imatrix warnings, the base-type mapping.

Deliberate differences, all in this file and none in llama.cpp:

- Explicit `--linattn/--fullattn/--embd/--output-type` override the hybrid
  profiles' pinned values instead of being discarded by them.
- The dense-hybrid attention pinning applies to `dense-hybrid-quality` too,
  matching upstream's `if` in `generate_config.sh`.
- `--estimate` also applies llama.cpp's real F32 rule (name-based) on top of
  upstream's size heuristic, so 1D and non-`weight` tensors are not counted as
  quantizable.

Known gaps, upstream's as well: `ssm_conv1d.weight` is not covered by the MoE
emitter (it falls to the base mixture), and the MTP-aware config patcher and
the full convert->imatrix->quantize pipeline are not ported.

## Why the anchoring matters

`llama-quantize` matches each tensor name against the config with
`std::regex_search`: **unanchored, first match wins** (`src/llama-quant.cpp`).
Two consequences the tests pin:

- An unanchored `output.weight` rule would capture every
  `blk.N.attn_output.weight` and silently mis-quantize the attention outputs.
  Every dense/opt pattern is anchored for this reason.
- The MoE emitter writes unanchored `blk.N.attn_q`, which also matches
  `blk.N.attn_qkv.weight`. Harmless today only because every MoE profile gives
  both the same type -- one profile that splits them ships a wrong quant with
  nothing in the log to show it. `test_no_conflicting_multi_match` replays
  every profile against four real tensor-name fixtures to keep that true.

## Licence

Ported and vendored from `localai-org/apex-quant`, MIT; see upstream
`LICENSE`. `tools/quantize/README.md` points here from the flags it documents.
