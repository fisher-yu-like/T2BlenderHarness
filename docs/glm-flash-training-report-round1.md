# glm-5.3-flash assistant-session 训练报告（Rounds 1-2 完成；Rounds 3-6 恢复路径）

日期：2026-08-30 · 分支：`glm-5.3-flash`（自 `codex/director-multi-entity-harness` @71ba36a 检出）·
工作树：`C:\Users\sy\Desktop\T2BlenderCode\.worktrees\glm-5.3-flash`

## 1. 状态总览

| 轮次 | 状态 | Attempt | Train 成功率 | Dev 成功率 | Train overall 均值 | Dev overall 均值 |
|---|---|---|---|---|---|---|
| Round 1 | ✅ 完成 | a1 + a2 | 15/20 → **20/20** | 15/20 → **20/20** | 62.0 → **63.1** | 51.9 → **60.2** |
| Round 2 | ✅ 完成 | a1 + a2 | 10/10 (a2) | 8/10 scored (a2) | **44.3** | **48.8** |
| Round 3-6 | ⏳ 待续 | — | — | — | — | — |
| Test 100 | 索引已建 | — | — | — | — | — |

Round-2 分数低于 Round-1 是预期行为：Round-2 的 train-02/dev-02 是 Human_Interaction 家族（人物交互），物理和空间复杂度远高于 Round-1 的 Camera_Motion 家族（单主体 + 相机运动），且小道具（phone/tie/brush）在 320×180 渲染分辨率下难以辨认。跨轮比较应在同 family 内进行。

## 2. 实现方式（不变）

- 100-case test 索引：`dataset/vbench2-agent-test-100-v1`（10 维度 × 10 prompt，verbatim，零重叠，自检通过）。
- Director 解释 + Blender 代码 + VLM 评审：全部由驱动 glm-5.3-flash 会话在环内实时生成（零 preauth、零模板）。
- 内循环重生成上限 2 次（用户指令）。

## 3. Round-1 详细结果

### 3.1 attempt-1 → attempt-2 配对

| Case | Prompt | a1 | a2 | Δ |
|---|---|---:|---:|---:|
| train-01-01 | Garden, zoom out. | 64.9 | 65.3 | +0.4 |
| train-01-02 | …orbits… clockwise. Garden. | 59.9 | 60.0 | +0.1 |
| train-01-03 | Pyramid, tilt down. | 72.4 | 72.4 | 0 |
| train-01-04 | Mount Fuji, zoom in. | 59.8 | 66.2 | +6.4 |
| train-01-05 | Mount Fuji, pan left. | 71.9 | 71.9 | 0 |
| train-01-06 | Blue Lagoon, tilt up. | 68.2 | 67.4 | -0.8 |
| train-01-07 | Blue Lagoon, static… | NR | 71.0 | new |
| train-01-08 | Table, pan left. | 45.4 | 45.6 | +0.2 |
| train-01-09 | Alhambra, zoom out. | 62.8 | 63.4 | +0.6 |
| train-01-10 | Alhambra, pan right. | 62.4 | 65.2 | +2.8 |
| dev-01-11 | Vase, tilt down. | 72.1 | 73.1 | +1.0 |
| dev-01-12 | …orbits… clockwise. Vase. | 59.7 | 59.7 | 0 |
| dev-01-13 | Burj Khalifa, pan left. | 41.7 | 53.9 | **+12.2** |
| dev-01-14 | Machu Picchu, tilt up. | 58.0 | 64.7 | +6.7 |
| dev-01-15 | …Machu Picchu, static… | NR | 69.3 | new |
| dev-01-16 | Forbidden City, pan left. | 46.9 | 59.6 | **+12.7** |
| dev-01-17 | Forbidden City, First-person…dolly. | 32.7 | 50.1 | **+17.4** |
| dev-01-18 | Laptop, pan right. | NR | 69.8 | new |
| dev-01-19 | Watch, zoom out. | NR | 66.5 | new |
| dev-01-20 | …orbits… clockwise. Watch. | NR | 59.4 | new |

## 4. Round-2 详细结果（attempt-2，哈希修复后）

| Split | Scored | Task 均值 | Realism 均值 | Overall 均值 |
|---|---:|---:|---:|---:|
| train-02 | 10/10 | 45.4 | 40.0 | 44.3 |
| dev-02 | 8/10 | 51.0 | 41.1 | 48.8 |

Round-2 失败的 2 个 dev case（02-14 fills cup / 02-19 brushes jacket）因 prompt 含弯引号 ' 触发 ensure_ascii 哈希分歧。修复已提交（`a412c9f` + `1a89e25`），但这两个 case 的内循环预算在修复前已消耗。

## 5. 训练过程中发现并修复的 harness 缺陷

| 提交 | Owner | 缺陷 → 修复 |
|---|---|---|
| `9c53d1c` | director_trajectory | handoff 缺 detach → 补全 |
| `9c53d1c` | evaluate 接线 | dataset_fingerprint 缺失 → 传入 |
| `aac67f7` | code_cache | 外层 attempt 缓存隔离 |
| `1276a10` | code_cache | 同计划重生成版本化 |
| `1a89e25` | real_artifacts | artifact gate plan hash ensure_ascii |
| `a412c9f` | director_contracts | content_hash ensure_ascii |

## 6. 恢复路径

### Round-3 (~6)

```bash
cd C:\Users\sy\Desktop\T2BlenderCode\.worktrees\glm-5.3-flash
uv run python scripts/train_real_harness.py --mode diagnostic-attempt --round 3 --attempt 1 \
  --dataset-root dataset/vbench2-agent-training-index-v1 \
  --readiness-report out/preflight-readiness-exact-v2.json \
  --round-root out/training/glm-flash-diagnostic-v1 \
  --blender-bin D:/blender/blender.exe --workers 4 --vlm-model gpt-5.6-luna \
  --provider-mode assistant \
  --markdown-path docs/t2blendercodeharness-glm-flash-training-memory-v1.md
```

期间用 `scripts/serve_current_case.py` 逐案响应 director + codegen 请求（每个 case 需要亲笔 spec + scene）。
每轮结束后运行 test-100：`scripts/run_batch_eval.py --dataset-root dataset/vbench2-agent-test-100-v1 --split test --run-root out/test-eval/round-N`。
最后 `--summarize` 出总分。

### Test-100 评测

```bash
uv run python scripts/run_batch_eval.py --dataset-root dataset/vbench2-agent-test-100-v1 \
  --split test --run-root out/test-eval/round-N --summarize
```

## 7. 诚实边界

- diagnostic_precalibration 模式，visual_scores_permitted=false。
- 视觉评审由驱动会话（glm-5.3-flash）担任 VLM，来源标记 codex_local_visual_review。
- 分数仅用于诊断；formal acceptance 仍被 training_allowed=false 阻止。

## 8. 完整保留索引

| 内容 | 位置 |
|---|---|
| 训练记忆表 | docs/t2blendercodeharness-glm-flash-training-memory-v1.md |
| Round-1/2 报告+分数 | out/training/glm-flash-diagnostic-v1/round-0{1,2}/ |
| Test-100 索引 | dataset/vbench2-agent-test-100-v1/ |
| 全部解释/代码/评审 | out/assistant-session/{preauth,live,scenes,authoring,reviews}/ |
| 提交历史 | git log glm-5.3-flash（baseline tag: baseline-glm-flash-v0） |
