# Real Blender Video Evaluation

- scoring mode: `real_blender_video_vlm`
- case count: `1`
- real videos: `1`
- video-scored cases: `0`
- mean final score: `None`
- mean task final score (separate channel): `None`
- mean artifact-only realism score (separate channel): `33.7371`

分数只来自真实 Blender 生成的 `proxy.mp4` 的采样帧，并标注 external_vlm 或 assistant_local_review 来源；artifact 不完整、VLM unavailable 或本地复核未完成的 case 不进入 mean final score。

| Case | Prompt | Proxy video | Deterministic | Video review | Task final score | Artifact-only realism | Artifact | Review status | Findings |
|---|---|---|---:|---:|---:|---:|---|---|---|
| vbench2-dev-01-20 | The camera orbits around in a clockwise direction. Watch. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/harness-rsi/out/training/glm-flash-six-rounds-v4/round-01/attempt-02/real/dev/vbench2-dev-01-20/proxy.mp4) | 100.0 | None | None | 33.7371 | complete | awaiting_assistant_review (assistant_local_review) | none |
