# Real Blender Video Evaluation

- scoring mode: `real_blender_video_vlm`
- case count: `10`
- real videos: `10`
- video-scored cases: `0`
- mean final score: `None`
- mean task final score (separate channel): `None`
- mean artifact-only realism score (separate channel): `34.0225`

分数只来自真实 Blender 生成的 `proxy.mp4` 的采样帧，并标注 external_vlm 或 assistant_local_review 来源；artifact 不完整、VLM unavailable 或本地复核未完成的 case 不进入 mean final score。

| Case | Prompt | Proxy video | Deterministic | Video review | Task final score | Artifact-only realism | Artifact | Review status | Findings |
|---|---|---|---:|---:|---:|---:|---|---|---|
| vbench2-train-02-01 | One person passes a ball to another. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/glm-5.3-flash/out/training/glm-flash-diagnostic-v1/round-02/attempt-01/real/train/vbench2-train-02-01/proxy.mp4) | 100.0 | None | None | 34.4226 | complete | awaiting_assistant_review (assistant_local_review) | none |
| vbench2-train-02-02 | One person puts a coat on another person. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/glm-5.3-flash/out/training/glm-flash-diagnostic-v1/round-02/attempt-01/real/train/vbench2-train-02-02/proxy.mp4) | 100.0 | None | None | 34.448 | complete | awaiting_assistant_review (assistant_local_review) | none |
| vbench2-train-02-03 | One person picks up something dropped by another. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/glm-5.3-flash/out/training/glm-flash-diagnostic-v1/round-02/attempt-01/real/train/vbench2-train-02-03/proxy.mp4) | 100.0 | None | None | 34.0668 | complete | awaiting_assistant_review (assistant_local_review) | none |
| vbench2-train-02-04 | Two people are building a puzzle together. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/glm-5.3-flash/out/training/glm-flash-diagnostic-v1/round-02/attempt-01/real/train/vbench2-train-02-04/proxy.mp4) | 100.0 | None | None | 33.7412 | complete | awaiting_assistant_review (assistant_local_review) | none |
| vbench2-train-02-05 | Two people are playing tug-of-war with a rope. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/glm-5.3-flash/out/training/glm-flash-diagnostic-v1/round-02/attempt-01/real/train/vbench2-train-02-05/proxy.mp4) | 100.0 | None | None | 34.1282 | complete | awaiting_assistant_review (assistant_local_review) | none |
| vbench2-train-02-06 | One person hands a phone to another person. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/glm-5.3-flash/out/training/glm-flash-diagnostic-v1/round-02/attempt-01/real/train/vbench2-train-02-06/proxy.mp4) | 100.0 | None | None | 33.895 | complete | awaiting_assistant_review (assistant_local_review) | none |
| vbench2-train-02-07 | Two people are hanging a picture on the wall. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/glm-5.3-flash/out/training/glm-flash-diagnostic-v1/round-02/attempt-01/real/train/vbench2-train-02-07/proxy.mp4) | 100.0 | None | None | 33.7412 | complete | awaiting_assistant_review (assistant_local_review) | none |
| vbench2-train-02-08 | One person pushes another person in a wheelchair. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/glm-5.3-flash/out/training/glm-flash-diagnostic-v1/round-02/attempt-01/real/train/vbench2-train-02-08/proxy.mp4) | 100.0 | None | None | 34.1114 | complete | awaiting_assistant_review (assistant_local_review) | none |
| vbench2-train-02-09 | One person brushes the hair of another person. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/glm-5.3-flash/out/training/glm-flash-diagnostic-v1/round-02/attempt-01/real/train/vbench2-train-02-09/proxy.mp4) | 100.0 | None | None | 34.0673 | complete | awaiting_assistant_review (assistant_local_review) | none |
| vbench2-train-02-10 | Two people are walking side by side down a hallway. | [proxy.mp4](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/glm-5.3-flash/out/training/glm-flash-diagnostic-v1/round-02/attempt-01/real/train/vbench2-train-02-10/proxy.mp4) | 100.0 | None | None | 33.6028 | complete | awaiting_assistant_review (assistant_local_review) | none |
