# T2Blendercodeharness 单实体五轮外循环训练计划 v7

状态：等待用户批准后开始真实渲染

日期：2026-08-26

## 目标与冻结项

目标是在 `dataset/trajectory-v3-single` 的 120 条单人物/单物体 prompt（50 train、50 dev、20 frozen test）上，从 round-01 做五轮 Harness 外循环进化。原始 `dataset/trajectory-v3-hard` 与旧 `out/training/six-rounds-real-v6` 只作为后续多实体升级与历史证据，不混入本轮统计；新运行根目录为 `out/training/single-five-rounds-v1`，记忆表为 `docs/t2blendercodeharness-six-round-training-memory-v7.md`。

冻结项：

- phase-1 prompt、split、oracle、数据集 fingerprint（`b86b25c3d4b94b2c12ec56a881acd05ffd0973df85f9b4297db4cbc15c1978a6`）；
- 本轮开始前的 evaluator v3 与评分公式；
- `blender/real_proxy_job.py` 当前真实化版本，除非后续 Harness proposal 明确选择 `proxy_renderer`；
- `D:\blender\blender.exe`、FPS、时长、采样策略和 artifact schema；
- test split 不能参与失败聚合、patch 选择或阈值拟合。

## 单 case 处理顺序

```text
Prompt
  → SceneContract
  → character/object trajectory + CameraPlan
  → Blender CLI 生成 .blend 和 animation PNG
  → 真实 MP4 组装与解码
  → artifact gate
  → deterministic + independent oracle
  → geometry/PNG realism evidence
  → 条件满足时一次 shared VLM review
  → 两个独立分数写入 report 和 Memory
```

Blender 失败时最多重试 2 次，每次都写 `render_attempts.json`。重试是执行可靠性机制，不是外循环 attempt；外循环每轮最多 5 次，每次仍只生成 10 train + 10 dev。

## 合并后的 evaluator 逻辑

合并的是执行链和帧准备，不是分数：同一个 real-run 同时产生 deterministic、geometry、PNG 和 task-VLM 证据；报告中明确保留两个 channel。

### 任务/轨迹通道

deterministic 检查 artifact、contract、telemetry、事件顺序、camera intent、support/contact、attachment、速度连续性和独立 oracle。只有 artifact 完整、MP4 可解码、deterministic 没有 hard failure 且 geometry gate 没有 hard failure 时才调用一次 VLM。VLM 使用同一组最多 8 个按事件对齐的真实帧，检查 prompt compliance、physical plausibility、camera coverage/innovation、character/object trajectory、event timing、temporal smoothness、visual clarity。

VLM 内部先分别用几何平均汇总相关维度：

```text
semantic = GM(prompt_compliance, physical_plausibility,
              object_trajectory, event_timing)
choreography = GM(camera_coverage, camera_innovation,
                  character_trajectory, temporal_smoothness)
task_vlm = .45 semantic + .45 choreography + .10 visual_clarity
```

任务通道的现有融合仍为：

```text
task_final = .20 deterministic + .80 task_vlm
```

若 VLM 不可用，不把 realism、deterministic 或本地猜测复制成 VLM 分数；写 `unavailable`。若采用 Codex 真实帧复核，写 `assistant_local_review`，只在有明确帧证据且 confidence ≥ 0.6 时作为 review 结果。

### 独立 realism 通道

`geometry_compliance` 只回答“真实 `.blend` 是否满足详细几何合同”，不回答“是否真实”。`visual_evidence` 只测真实 PNG 的可读采样率、分辨率、前景覆盖、边缘结构和时间变化。

```text
G = .20 coverage + .20 topology_detail
  + .20 non_primitive_representation
  + .10 semantic_integrity + .30 structural_detail

V = .20 availability + .15 resolution + .25 foreground
  + .20 edge_structure + .20 temporal_change

A_raw = .60 G + .40 V
artifact_only_proxy = min(80, .80 A_raw)
```

`.80` 为未观测的语义、物理、遮挡、接触、人物动作和轨迹可见性保留空间。geometry hard fail 将 `G` 限制到 20。没有 shared review 时只能记录 `artifact_only_proxy`，不能冒充 `realism_score`。

有 shared review 时，VLM/local Codex 是真实性主体：

```text
realism_score = .15 × G + .15 × V + .70 × realism_review
```

`task_score` 与 `realism_score` 不相加。

两个分数不相加、不互相替代。接受使用双门：任务通道 train 严格上升且 dev 不下降；如果 proposal 的 owner 是 `proxy_renderer` 或 realism 目标，则再单独要求 artifact-only realism 均值提升。一个通道的提升不能抵消另一个通道的 hard regression。

## 五轮调度

每轮先从对应 train family 和 dev family 取 10+10 case，完成 attempt-01；随后根据 train 失败聚合只提出一个 Harness owner 的 patch，再按最多 5 次 attempt 重跑。每轮结束生成该轮累计 train/dev 的整体评测，共 120 个视频。

| round | train family | dev family | attempt 上限 | round-end overall |
|---:|---|---|---:|---:|
| 1 | hard-01 | hard-07 | 5 | 120 |
| 2 | hard-02 | hard-08 | 5 | 120 |
| 3 | hard-04 | hard-09 | 5 | 60 |
| 4 | hard-05 | hard-10 | 5 | 80 |
| 5 | hard-06 | hard-11 | 5 | 100 |

每轮整体评测累计 20、40、60、80、100 个 case；保守上限为 `5 × (5 × 20 + 100) = 1000` 个真实视频。实际数量按每轮在接受/拒绝后是否继续 attempt 记录，不能预填满。

## 单组件进化规则

失败按 `failure_id + owner + category + severity + root_cause_id` 去重，只从 train 的重复模式产生 proposal。允许的 owner 为 `scene_parser`、`trajectory_planner`、`camera_planner`、`blender_executor`、`proxy_renderer`、`evaluator`。一次 patch 只能包含一个 owner；不得同时修改另一个 Harness 组件、数据集标签、oracle 或评分公式来换分。

每个 candidate 必须保存 parent/candidate Harness fingerprint、patch 文件、发现的问题、修复方法、attempt 结果、paired dev、累计 overall dev 和处理结论。接受条件：

```text
train_after > train_before
paired_dev_after >= paired_dev_before
overall_dev_after >= overall_dev_before
无 hard dev regression
```

若不满足，记录 rejected/rollback，不把 candidate 进入下一轮。若重复失败不具备跨样本证据，记录 `no_patch`。

## Memory 交付

每个 split 完成后立即更新 Markdown；每个 case 一行，至少包含：轮数、完整 prompt、真实 `proxy.mp4` 绝对地址、task 分数、artifact-only realism 分数、review 来源/置信度、检测到的 Harness 问题、severity/root cause、渲染重试次数、单组件修复位置和方法、修复前后提升/下降、自然语言处理结论。未渲染只能写 `NOT_RENDERED` 和原因。

每轮还保存：

```text
round-XX/attempt-YY/real/{train,dev}/
round-XX/overall/real/{train,dev}/
round-XX/attempt_report.json
round-XX/overall_report.json
round-XX/patch_manifest.json
memory/harness_updates.jsonl
```

## 开始前验收

批准后依次执行：

1. 用单实体数据集生成 v7 protocol，核对 fingerprint、5 轮、每轮 10+10、最多 5 次和 1000 上限；
2. 运行全量 pytest、skill capability check 和 1 case real Blender smoke；
3. 运行 round-01 attempt-01；完成 deterministic/geometry/PNG 后，根据 VLM endpoint 状态选择 external VLM 或 Codex `assistant_local_review`，不造分；
4. 立刻更新 v7 Markdown，再由 train 失败模式决定是否提出一个组件 patch；
5. 通过 paired train/dev gate 后才继续该轮或进入下一轮；
6. 五轮结束才对冻结 test 做一次盲评，并单独输出 task 与 realism 两条曲线。
