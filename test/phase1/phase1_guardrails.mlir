// RUN: ! rg -n 'pto\\.(load|abs|store)\\b' %S/*.mlir

// This is a shell-only lit test that keeps the Phase 1 fixture corpus free of
// legacy pseudo-op spellings.
