# Real Run Report

The corrected real harness was evaluated through calibration, train, and dev slices with connected Blender MCP.

- Calibration (`out/real/calibration-v4`): 10/10 complete, deterministic 10/10 pass.
- Train (`out/real/train-v1`): 20/20 complete, deterministic 20/20 pass.
- Dev (`out/real/dev-v1`): 10/10 complete, deterministic 10/10 pass.
- PNG animation: 240 frames for standard cases; 480 frames for long-composition cases.
- Host MP4 assembly: 40/40 runs have `proxy.mp4`; all sampled PNG frames are readable.
- Semantic proxy validation: telemetry now records the generated entity kind; the evaluator hard-fails kind mismatches. An earlier calibration render exposed `table` being generated as a prop, which was fixed before v4/train/dev.
- Outer-loop report: `out/real/outer_loop_v1/optimization_report.json` records `no_patch`; no repeated train failure exists, so the one-owner patch gate correctly did not modify the Harness.
- VLM: the adapter was exercised on deterministic-pass calibration outputs but returned `unavailable` because the configured endpoint returned HTTP 403/1010. No VLM score is used for Harness evolution until a compliant tenant-approved endpoint is configured.
