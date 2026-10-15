# HANDOVER.md - Garuda

Append-only. Newest entry last; each entry names what changed and what was
measured.

## 2026-10-15 - the fork exists

`Rin247/Garuda` is a fork of `ggml-org/llama.cpp`, cloned to
`C:\Projects\Garuda` at upstream `843e7dcc1` ("spec : add lilicorr support
(#30506)").

Naming, for the record: the fork was first created as `Rin247/llama.cpp`,
renamed to `Rin247/Cheetah`, and the rename to `Garuda` first failed with 422
(name already existed on the account - a pre-existing empty repo). The owner
granted the `delete_repo` scope, the placeholder was deleted, and the rename
succeeded. Both old names redirect to `Garuda`.

## 2026-10-15 - APEX is in the tree

Owner: *"I also want to bake in apex quants into it by default."* Upstream
`localai-org/apex-quant` (MIT) is two bash scripts over stock `llama-quantize`;
the port is one stdlib-only Python file at `tools/quantize/apex/apex.py`, and
the tensor-name fixtures plus the bits-per-weight table are vendored from the
same repo. Default profile `balanced`; layer count read from the GGUF's own
`block_count`; `--estimate` predicts the output size from the tensor inventory.

Second sync the same day, after the owner noted upstream had moved since the
first read: the current `quantize.sh` and `generate_config.sh` were byte-equivalent
to what had been ported, but the repo had grown the sensitivity-fitted `opt-*`
levels, `estimate_config_size.py`, and a collision test with real fixtures -
all three ported. Three upstream behaviours were reproduced deliberately
rather than "fixed": the `--quiet` size prints three decimals, the F32 floor is
a size heuristic on top of llama.cpp's name rule, and `dense-hybrid-quality`
shares `dense-hybrid`'s attention pinning. Deviations are listed in
`tools/quantize/apex/README.md`.

Measured: 53 tests, exit 0, 7.3 s. Three of them caught real defects in the port
during development - a `mid_exp` KeyError in `custom` profile resolution, a
struct format mismatch reading UINT32 metadata, and warnings that sat behind the
`llama-quantize not found` guard and so never printed. A fourth caught the test
suite itself: the collision sweep recompiled 680 regexes against 700 tensor
names per profile (210 s), fixed by precompiling and emitting in-process.

## 2026-10-15 - estate identity and cleanup

Owner: *"I wouldn't mind cleaning up both Garuda and Agni tbh"*, then the wider
steer - Prometheus uses Agni for torch and Garuda for llama, built by GitHub
Actions or manually, *"this is why I wanna clean up both forks so we cut dead
ends/fat"*.

Done here: the four project documents, the estate section appended to upstream's
`AGENTS.md` (its content untouched - it says itself that its rules are for
`ggml-org/llama.cpp` and not for a fork), `.kilo/` ignored, the `*.bat`
negations, and the regenerable `__pycache__` removed from both projects.

History: the fork's 11,641 upstream commits and 8,837 remote branches were cut
to a single commit on `main`, and the `upstream` remote was configured for
on-demand fetches. `.git` went from 453.7 MB to the size of the tree. The
consequence, recorded rather than hidden: there is no shared ancestor with
upstream, so updating means taking upstream's tree and re-applying the
fork-local file list in `PROJECT.md`.

## 2026-10-15 - while the owner builds

The owner began an sm_89 build of Agni so tests could run against it. No
compiles were started from this session and nothing under `build-CUDA` was
touched. The conformance gate was attempted once and exceeded a 10-minute
window with a compile live on the machine; it is re-run after the build rather
than competing with it.
