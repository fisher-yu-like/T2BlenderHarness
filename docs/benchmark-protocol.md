# Proxy Benchmark Protocol

The benchmark is versioned as `benchmark-v1` and runs the frozen `dataset/manifest.jsonl` through exactly one of `train`, `dev`, or `test` from `dataset/splits.json`.

Each cache key binds the prompt, Harness version, evaluator version, backend, seed, and white-proxy render settings. Reports contain per-case status, score, failure IDs, cache key, aggregate pass rate, and mean score. The fake backend is deterministic and is used for local reproducibility checks; a real Blender backend must record its version separately.

The test split is frozen for milestone comparisons. It must not be used to select a Harness patch, tune evaluator weights, or change labels during an active comparison. A comparison is valid only when Harness, evaluator, backend, seed, and render settings are recorded in the report.
