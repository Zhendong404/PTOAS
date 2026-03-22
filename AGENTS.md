# Repository Guidelines

## Project Structure & Module Organization
`include/PTO` and `lib/PTO` contain the core dialect, transforms, and lowering logic. CLI tools live under `tools/` (notably `tools/ptoas`). Python bindings and sample builders are under `python/` and `test/samples/`. CMake entry points are in `CMakeLists.txt` and `cmake/`. Design notes and evolving specs live in `docs/`, `a5vm.md`, and `vpto-spec.md`. Treat `.planning/` as workflow state, not product code.

## Build, Test, and Development Commands
Source the environment before building or running samples:

```bash
source env.sh
bash do_cmake.sh --llvm "$LLVM_ROOT"
cmake --build build -j
```

- `source env.sh`: exports LLVM/PTOAS paths and Python bindings.
- `bash do_cmake.sh --llvm "$LLVM_ROOT"`: configures the in-tree `build/` directory.
- `cmake --build build -j`: builds `ptoas`, libraries, and bindings.
- `./test/samples/runop.sh -t Abs`: compiles one sample family.
- `./test/samples/runop.sh all`: runs the sample sweep.
- `bash test/samples/run_a5vm_acceptance_checks.sh`: runs focused A5VM regression checks.

## Coding Style & Naming Conventions
Use C++17 and existing MLIR/LLVM idioms. Match nearby style: 2-space indentation in TableGen, 2-4 spaces in C++ depending on file, no unnecessary comments, and keep code ASCII unless the file already uses Unicode. Prefer descriptive lowering helpers such as `lowerTLOAD`, `build...Scope`, and `extract...Contract`. New sample files in `test/samples/` should use lowercase snake case unless mirroring an existing sample family.

## Testing Guidelines
Validate changes with the smallest relevant sample first, then rerun broader coverage. For backend work, prefer `--pto-backend=a5vm --a5vm-print-ir` and inspect raw A5VM IR before textual HIVM emission. Keep regression checks in `test/samples/runop.sh` aligned with the active backend output format. If a sample is intentionally unsupported, mark it explicitly as `SKIP` or `XFAIL` with a concrete reason.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects, for example: `Implement and validate A5VM backend updates`. Follow that style. Keep each commit scoped to one coherent change. PRs should include:
- the problem statement and affected lowering path,
- exact validation commands run,
- any samples newly passing, skipped, or intentionally deferred,
- IR snippets or output paths when the change is backend-facing.

## Configuration Notes
Do not rely on ad hoc build directories in `/tmp`; this repo builds in `build/`. Avoid committing local environment helpers unless they are intentionally shared project scripts.
