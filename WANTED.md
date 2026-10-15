# WANTED.md - Garuda

The estate-wide `WANTED.md` at the root is the single authority for what the
owner has asked for across all projects. **Nothing here restates it.** This
file holds only what bears on Garuda.

## Status

| # | What | State | Notes |
|---|---|---|---|
| 1 | Fork llama.cpp to `Rin247/Garuda` | **done** | cloned 2026-10-15 from `ggml-org/llama.cpp` at `843e7dcc1`; the rename history (`llama.cpp` -> `Cheetah` -> `Garuda`) is in `HANDOVER.md` |
| 2 | Estate identity | **done** | this file, `PROJECT.md`, `HANDOVER.md`; the estate section appended to `AGENTS.md`; `.kilo/` ignored; the `*.bat` negations upstream's ignore needs |
| 3 | Bake in APEX quants by default | **done, tested** | Owner 2026-10-15. `tools/quantize/apex/` - profile emitter, quantize driver, size estimator; default profile `balanced`; layer count read from the GGUF's own `block_count`, not a hardcoded 40. 53 tests, exit 0 |
| 4 | Be Prometheus's llama.cpp | **wanted, not started** | Owner 2026-10-15: *"we basically want Prometheus to use Agni and Garuda for torch/llama"*. The integration shape (vendored source, submodule, or prebuilt artefact) is undecided and is the first decision this row needs |
| 5 | Lean build: CUDA + CPU only, sm_89 | **wanted, not started** | One configure line with `GGML_CUDA` on and every other backend off, `GGML_CCACHE=OFF` on the line itself (root `AGENTS.md` 25), output to `build/Release` (42), `-j 8`. The machine has no ROCm, no oneAPI and no WSL |
| 6 | Build for us: GitHub Actions or manual | **wanted, not started** | Owner 2026-10-15: *"we just have github build for us or we do it manually"*. A windows runner can build this tree; whether the artefact is a wheel-shaped package or the DLLs is row 4's decision |
| 7 | One measured win | **wanted, not started** | The owner's framing: make Garuda lighter and faster. One kernel or one build-size change, measured before and after; `apex.py --estimate` is already the measuring tool for the size half |
| 8 | Verb scripts | **partial** | `.gitignore` now allows them; the scripts themselves are not written. `test.bat` runs the APEX suite today, `build.bat` is row 5's configure line |

## Deferred by the owner

* APEX is *"just a bonus since we wanna support that going forward"* - the
  default path is landed and tested, the measurement work behind the dense
  profiles stays upstream's.

## On the record, not on the list

* **The conformance gate scopes Garuda as `desktop`** because it has a root
  `CMakeLists.txt` (`tools/estate_conformance.py:278`), which is tree evidence,
  not a decision. Rows about a Qt app shape (exe name, data root, render
  harness) will fail against a toolchain fork; that is a taxonomy gap recorded
  in `PROJECT.md`, not a defect in Garuda to fix by editing root tools.
* **`tools/quantize/README.md` carries one fork-local section.** It is upstream's
  file; the added lines point at `apex/` and are the only upstream file changed
  beside the two `.gitignore` fences.
