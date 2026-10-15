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
