# PROJECT.md - Garuda

## Now

Garuda is a fork of `ggml-org/llama.cpp` (MIT) at upstream `843e7dcc1`. It
exists for two reasons, in order:

1. **It is Prometheus's llama.cpp.** Owner, 2026-10-15: *"we basically want
   Prometheus to use Agni and Garuda for torch/llama."* The integration shape -
   vendored source, submodule, or prebuilt artefact fetched from a build - is
   undecided and is the first open decision (`WANTED.md` row 4).
2. **APEX quantization is baked in.** `tools/quantize/apex/` carries the
   MoE-aware mixed-precision allocator so it is the default path, not an
   optional add-on. Ported from `localai-org/apex-quant` (MIT), synced
   2026-10-15.

**History is one commit by design** (owner, 2026-10-15). Upstream's 11,641
commits and 8,837 branches were cut: a CI checkout or a fresh clone of this fork
pulls 171 MB instead of 454 MB, and the fork's diff is small enough to carry in
prose. The `upstream` remote points at `ggml-org/llama.cpp` and is fetched on
demand; updating means taking upstream's tree and re-applying the fork-local
files listed below, not rebasing onto a shared ancestor that no longer exists.

## The fork-local diff

Everything that is not upstream lives here, and this list is the whole of it:

| path | what |
|---|---|
| `tools/quantize/apex/` | the APEX port: `apex.py`, `test_apex.py`, `README.md`, `tests/fixtures/` |
| `tools/quantize/README.md` | one appended section pointing at `apex/` |
| `.gitignore` | `.kilo/`, and negations re-including the estate verb scripts that upstream's `*.bat` rule ignores |
| `AGENTS.md` | the estate section appended below upstream's instructions, which are untouched |
| `PROJECT.md`, `HANDOVER.md`, `WANTED.md` | this project's own documents |

No llama.cpp or ggml source file is modified. `llama-quantize`'s
`--tensor-type-file` is upstream's own flag (tools/quantize/quantize.cpp:435);
APEX is a recipe over it, which is why the port needs no patched build.

## APEX in one paragraph

APEX assigns a type per tensor role and per layer position: routed experts take
the hardest cut, the shared expert stays at Q8_0 because its weights are
heavy-tailed, and the edge layers keep precision because they do embedding
alignment and logit generation. `apex.py` emits the `--tensor-type-file`, drives
stock `llama-quantize`, reads the layer count from the GGUF's own `block_count`
(upstream's default assumes one specific 40-layer model), and can predict the
output size from the tensor inventory before a quantize run. Profiles: the MoE
tiers, the dense/hybrid arms, the sensitivity-fitted `opt-*` allocations.
Details, differences from upstream and the known gaps are in
`tools/quantize/apex/README.md`.

## The other quantiser in the estate

The Forge (`C:\Tools\Forge`, outside the estate root) is the second one, and
it does a different job. APEX quantises **GGUF** models for stock
`llama-quantize`; the Forge quantises **diffusers-layout** models - Anima,
Qwen-Image 2.1, FLUX 3 Image Klein, MOSS-Image - into torchao float8, INT8,
FP4 and INT4 packages, one directory per format under
`C:\Tools\Forge\quant_lab`. The two do not overlap: APEX's unit is a tensor
role inside a GGUF block, the Forge's unit is a component directory in a
diffusers repository.

Measured 2026-10-20, and it is why the image models are not this fork's
business: **llama.cpp's diffusion path is language-model shaped.**
`examples/diffusion/README.md` names dream7b, llada-8b and RND1-Base, and no
"flux" appears anywhere in the tree. FLUX 3 is a denoiser-and-encoder image
model, so there is no GGUF conversion for it and APEX has nothing to
allocate. The image models go through diffusers and the Forge, and the work
Prometheus's Studio does with them.

One finding from that route is worth recording here, because it is the shape
of the problem APEX would meet if a GGUF image model ever arrived: **the
Forge's packages are not loadable by diffusers as they stand.** The
quantised tensors are torchao `float8_weight_only` - a float8 weight and a
separate float32 scale per tensor - and the package's `config.json` carries
no `quantization_config`, so diffusers reads the float8 blobs as if they were
bf16. The vendors' own ComfyUI `int8-convrot` files are no better inside
diffusers: 0.42 has no convrot and no SVDQuant loader. Both formats are one
metadata line and one dequantiser away from working, and neither is the GGUF
path, so neither touches this fork.

## Toolchain

| tool | path |
|---|---|
| cmake | `C:\Dev\cmake\bin\cmake.exe` (4.4.3) - the machine's PATH has the devkitPro MSYS2 cmake first and it fails at `Detecting CXX compiler ABI info` |
| ninja | `C:\Dev\Qt\Tools\Ninja` |
| MSVC 2022 | the Qt 6.12 msvc2022_64 toolchain environment, which must be in the same command as the configure (root `AGENTS.md` 24) |
| python | 3.15.0 (`%LOCALAPPDATA%\Programs\Python\Python315`) for the APEX tool and its tests |
| GPU | RTX 4060, sm_89, CUDA 13.4 |

**The MSYS2 shadowing is fixed at the source.** The devkitPro MSYS2 `usr\bin`
used to be first on the machine PATH, so a bare `git`, `cmake`, `make`, `tar`,
`awk`, `sed`, `perl` resolved to the unix-flavoured MSYS build, and `cmd`
itself resolved to a script named `cmd` in that directory - every `cmd /c` in a
shell silently ran the wrong program. The estate rule is absolute paths, and
every script here follows it, which is what kept this tree building while the
trap stood.

**Resolved 2026-10-15** (owner's decision): the machine PATH no longer has
`C:\Dev\devkitPro\msys2\usr\bin`, and the pinned cmake (`C:\Dev\cmake\bin`) plus
ninja (`C:\Dev\Qt\Tools\Ninja`) now sit at the front, so a bare name finds the
right tool. Verified in a fresh environment: git resolves to
`C:\Program Files\Git\cmd` (2.55.0), cmake to 4.4.3, and `cmd` to System32.
devkitPro's console toolchains are untouched - they are launched through
`msys2_shell.cmd`, which builds its own PATH (`/opt/devkitpro/tools/bin` first),
and never inherited the machine one. **Keep the absolute paths anyway**: they
are the estate rule, the backup of the old PATH is in the agent scratch
directory, and the ordering can be changed again.

No ROCm, no oneAPI, no WSL on this machine, so the lean build is CUDA + CPU and
nothing else (`WANTED.md` row 5).

## Estate deviations, each with its reason

- **No single `<AppName>.exe` (29).** llama.cpp is a toolchain fork: it builds
  `llama-cli`, `llama-quantize`, `llama-server` and the rest. One exe name does
  not apply; the artefact of record for APEX is `llama-quantize`.
- **No `~/.Garuda` data root (15a).** llama.cpp stores no settings, catalogues
  or state; the rule scopes to an app that does.
- **No render harness (22/23).** There is no GUI to open PNGs of.
- **CTest (26) applies and is upstream's**: `enable_testing()` and `add_test()`
  are in the tree, so the suite lands in `build/Release` and `ctest` runs it -
  `ctest` is not on PATH here, it is
  `C:\Dev\Qt\Tools\CMake_64\bin\ctest.exe` (root `AGENTS.md`, `QUIRKS.md` 22).
- **The conformance gate scopes Garuda as `desktop`** from tree evidence (a root
  `CMakeLists.txt`, `tools/estate_conformance.py:278`), not from a decision. Its
  exe-name, data-root and render rows will fail against a toolchain fork; that
  is the taxonomy gap `QUIRKS.md` 28 records, and it is reported rather than
  fixed by editing root tools.

## The `*.bat` trap

Upstream's `.gitignore:4` ignores every `.bat` in the tree. The estate's verb
scripts are `.bat` files at the project root (root `AGENTS.md` 34), so without
the negations in `.gitignore` a `git add build.bat` would silently stage
nothing. The negations are in the diff table above.

## Third-party

| checkout | licence | read by | what was taken |
|---|---|---|---|
| `ggml-org/llama.cpp` | MIT | Garuda (parent) | the whole tree; this is a fork of it |
| `localai-org/apex-quant` | MIT | Garuda | read 2026-10-15; the two recipe scripts ported to `apex.py`, the bits-per-weight table, and four tensor-name fixtures vendored into `tools/quantize/apex/tests/fixtures/`. Facts and allocations, no linked code |

## Verified

- `python tools/quantize/apex/test_apex.py` - 53 tests, exit 0, 2026-10-15.
  Covers every profile's line count and band boundaries, every emitted type
  checked against ggml's own `type_traits` table, llama-quantize's
  first-match-wins matching replayed against four real tensor-name fixtures for
  every profile (no conflicting multi-match), the GGUF metadata and tensor
  readers against synthesised files including a v1 header, and the size
  estimator's arithmetic.
- **Not yet verified:** anything that needs a build - the lean configure line,
  CTest, and the conformance gate. The gate was started on 2026-10-15 and
  exceeded a 10-minute window while a compile was live on this machine; a
  measurement taken during a build is not evidence, so it is re-run after.

## The x86-64 floor

Native code here is built for **x86-64-v3** (AVX2) and for **v4** (AVX-512),
and nothing older is supported - `WANTED.md` row 34 is the authority. The
build carries it in `CMakeLists.txt:6`: `X86_ISA` is a CMake cache variable, `v3` by
default, and `-DX86_ISA=v4` builds the AVX-512 variant - which does not run
on this machine, because the guest CPU has no AVX-512 (measured 2026-10-20).
