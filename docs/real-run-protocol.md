# Real Run Protocol

1. `scripts/prepare_real_jobs.py --split calibration|train|dev` creates immutable jobs.
2. The connected Blender MCP executes each `blender_job.py`; Blender writes `.blend`, PNG animation frames, sampled frames, telemetry, and an updated manifest.
3. `scripts/evaluate_real_runs.py` assembles `proxy.mp4` with host `imageio-ffmpeg`, validates artifacts, and runs deterministic rules.
4. Only deterministic-pass runs enter `scripts/evaluate_real_videos.py` for VLM scoring. An unavailable VLM result is never converted into a numeric training score.
5. `scripts/run_real_outer_loop.py` aggregates real train/dev reports and calls `MetaHarnessOptimizer`; a patch may own one Harness component only. The patch is accepted only when train improves and dev does not decline.

The current connected Blender is FFMPEG-less, so video encoding is deliberately host-side. The current VLM environment points to a non-tenant-approved proxy; those calls are recorded as `unavailable` until a compliant endpoint is configured.
