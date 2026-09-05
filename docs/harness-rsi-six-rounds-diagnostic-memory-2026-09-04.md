# Real Blender Video Evaluation

- scoring mode: `real_blender_video_vlm`
- case count: `10`
- real videos: `4`
- video-scored cases: `0`
- mean final score: `None`
- mean task final score (separate channel): `None`
- mean artifact-only realism score (separate channel): `32.8514`

分数只来自真实 Blender 生成的 `proxy.mp4` 的采样帧，并标注 external_vlm 或 assistant_local_review 来源；artifact 不完整、VLM unavailable 或本地复核未完成的 case 不进入 mean final score。

| Case | Prompt | Proxy video | Deterministic | Video review | Task final score | Artifact-only realism | Artifact | Review status | Findings |
|---|---|---|---:|---:|---:|---:|---|---|---|
| vbench2-train-01-01 | Garden, zoom out. | NOT_RENDERED: `D:\harness-rsi-training\diagnostic-six-rounds-latest-model-20260904\round-01\attempt-01\real\train\vbench2-train-01-01\proxy.mp4` | None | None | None | None | incomplete | not_run (none) | none |
| vbench2-train-01-02 | The camera orbits around in a clockwise direction. Garden. | NOT_RENDERED: `D:\harness-rsi-training\diagnostic-six-rounds-latest-model-20260904\round-01\attempt-01\real\train\vbench2-train-01-02\proxy.mp4` | None | None | None | None | incomplete | not_run (none) | none |
| vbench2-train-01-03 | Pyramid, tilt down. | NOT_RENDERED: `D:\harness-rsi-training\diagnostic-six-rounds-latest-model-20260904\round-01\attempt-01\real\train\vbench2-train-01-03\proxy.mp4` | None | None | None | None | incomplete | not_run (none) | none |
| vbench2-train-01-04 | Mount Fuji, zoom in. | NOT_RENDERED: `D:\harness-rsi-training\diagnostic-six-rounds-latest-model-20260904\round-01\attempt-01\real\train\vbench2-train-01-04\proxy.mp4` | None | None | None | None | incomplete | not_run (none) | none |
| vbench2-train-01-05 | Mount Fuji, pan left. | NOT_RENDERED: `D:\harness-rsi-training\diagnostic-six-rounds-latest-model-20260904\round-01\attempt-01\real\train\vbench2-train-01-05\proxy.mp4` | None | None | None | None | incomplete | not_run (none) | none |
| vbench2-train-01-06 | Blue Lagoon, tilt up. | NOT_RENDERED: `D:\harness-rsi-training\diagnostic-six-rounds-latest-model-20260904\round-01\attempt-01\real\train\vbench2-train-01-06\proxy.mp4` | None | None | None | None | incomplete | not_run (none) | none |
| vbench2-train-01-07 | The camera movement is static. Blue Lagoon, static shot, the camera is fixed. | [proxy.mp4](D:/harness-rsi-training/diagnostic-six-rounds-latest-model-20260904/round-01/attempt-01/real/train/vbench2-train-01-07/proxy.mp4) | 100.0 | None | None | 37.2 | complete | not_run (none) | none |
| vbench2-train-01-08 | Table, pan left. | [proxy.mp4](D:/harness-rsi-training/diagnostic-six-rounds-latest-model-20260904/round-01/attempt-01/real/train/vbench2-train-01-08/proxy.mp4) | 100.0 | None | None | 30.9733 | complete | not_run (none) | none |
| vbench2-train-01-09 | Alhambra, zoom out. | [proxy.mp4](D:/harness-rsi-training/diagnostic-six-rounds-latest-model-20260904/round-01/attempt-01/real/train/vbench2-train-01-09/proxy.mp4) | 100.0 | None | None | 31.9691 | complete | not_run (none) | none |
| vbench2-train-01-10 | Alhambra, pan right. | [proxy.mp4](D:/harness-rsi-training/diagnostic-six-rounds-latest-model-20260904/round-01/attempt-01/real/train/vbench2-train-01-10/proxy.mp4) | 100.0 | None | None | 31.263 | complete | not_run (none) | none |
