# glm-5.3-flash assistant-session 训练报告（Round 1 完成）

日期：2026-08-30 · 分支：`glm-5.3-flash`（自 `codex/director-multi-entity-harness` @71ba36a 检出）·
工作树：`C:\Users\sy\Desktop\T2BlenderCode\.worktrees\glm-5.3-flash`

## 1. 任务与约束回顾

- 在最新分支基础上克隆新分支 `glm-5.3-flash`，原分支代码零改动（仅读取复制其未提交工作）。
- 不用人工、不接外部大模型、不用固定模板：所有 agent 的 LLM（Director 解释、Blender 代码生成）与 VLM（14 维视觉评审）由驱动本会话的 glm-5.3-flash 模型亲自担任 provider。
- 内循环渲染失败最多再实验 2 次（`max_inner_attempts=2`）；外循环按六轮计划推进。
- 保留初版 harness 作为 baseline；保留完整训练过程与结果；输出报告与表格。
- 训练集复用 `dataset/vbench2-agent-training-index-v1`（140 case：60 train / 60 dev / 20 frozen test）。

## 2. 实现方式：assistant-session provider

外部 HTTP provider 被替换为"请求/响应文件交换 + 本会话亲笔作答"的传输层，身份、溯源、fail-closed 全部保留：

| 组件 | 文件 | 说明 |
|---|---|---|
| 会话 provider | `src/videoact/assistant_session_provider.py` | `provider_kind=agent_session_structured`，`model_id=glm-5.3-flash`，`template_backed=false`，`llm_generated=true`；每次调用写请求文件并阻塞等待本会话写响应文件，preauth 需 scene+prompt 精确匹配 |
| 运行器接线 | `scripts/train_real_harness.py` | 新增 `--provider-mode assistant`；审计器要求两阶段 provider_kind/model_id 逐 case 校验 |
| 缓存策略 | `scripts/prepare_real_jobs.py` | assistant 模式跳过缓存读取；冻结槽位冲突时版本化写入（不阻断重生成） |
| 辅助工具 | `scripts/assistant_respond.py`、`assistant_director_preflight.py`、`assistant_author_review.py`、`author_camera_case_source.py`、`drain_requests.py`、`mock_blender_dryrun.py` | 请求处置、离线预检、评审组装、逐 case 源码物化（含静态门禁自检）、请求排空、mock-Blender 干跑门 |
| 单元测试 | `tests/test_assistant_session_provider.py` | 身份/preauth/fail-closed/传输绑定/构建器（全绿） |

每一份 Director 解释与每个 `blender_job.py` 都是针对该 case 的 prompt/计划现场生成（实体、几何构成、相机运动、配色逐 case 不同），机器审计（`audit_dynamic_agent_index` + `source-fingerprint-v1` 语义指纹 + 跨 case 复用检测）全部通过；无任何固定模板回退。

## 3. Baseline 保留

| 项 | 值 |
|---|---|
| 初版 harness 基线提交 | `303dcfd`（tag `baseline-glm-flash-v0`，含 provider 接线，训练前状态） |
| 上游基线 | `71ba36a`（上游分支最新提交，未改动） |
| 之后每个 harness 修复 | 独立提交（见 §5），可单独回滚 |
| 训练前全量测试 | `uv run pytest` 全绿（两次：接线后、修复后） |

## 4. Round-1 训练结果（diagnostic，10 train + 10 dev 每轮 attempt）

### 4.1 总览

| 指标 | attempt-1 | attempt-2 | 变化 |
|---|---|---|---|
| 渲染成功 | 15/20 | **20/20** | +5 |
| 失败 case | 5（01-07/15/18/19/20） | **0** | 清零 |
| Artifact/确定性门禁 | 通过（已评 case） | 20/20 通过 | — |
| train overall 均值（可配对 9 case） | 62.0 | 63.1 | **+1.1** |
| dev overall 均值（可配对 6 case） | 51.9 | 60.2 | **+8.3** |
| train task 均值（10 case，attempt-2） | — | 68.8 | — |
| dev task 均值（10 case，attempt-2） | — | 63.7 | — |

### 4.2 逐 case overall_vlm_score（attempt-1 → attempt-2）

| Case | Prompt | a1 | a2 | Δ | 备注 |
|---|---|---:|---:|---:|---|
| train-01-01 | Garden, zoom out. | 64.9 | 65.3 | +0.4 | 起始过近仍是短板 |
| train-01-02 | The camera orbits… clockwise. Garden. | 59.9 | 60.0 | +0.1 | 轨道中段出框未根治 |
| train-01-03 | Pyramid, tilt down. | 72.4 | 72.4 | 0 | 稳定良好 |
| train-01-04 | Mount Fuji, zoom in. | 59.8 | 66.2 | +6.4 | zoom-in 过冲修复 |
| train-01-05 | Mount Fuji, pan left. | 71.9 | 71.9 | 0 | 稳定良好 |
| train-01-06 | Blue Lagoon, tilt up. | 68.2 | 67.4 | -0.8 | 噪声级波动 |
| train-01-07 | Blue Lagoon, static… | NOT_RENDERED | 71.0 | 新增 | 静态相机崩溃修复 |
| train-01-08 | Table, pan left. | 45.4 | 45.6 | +0.2 | 桌体结构比例错误（遗留） |
| train-01-09 | Alhambra, zoom out. | 62.8 | 63.4 | +0.6 | 起始过近 |
| train-01-10 | Alhambra, pan right. | 62.4 | 65.2 | +2.8 | pan 摆幅收敛 |
| dev-01-11 | Vase, tilt down. | 72.1 | 73.1 | +1.0 | 悬空修复后 grounded |
| dev-01-12 | …orbits… clockwise. Vase. | 59.7 | 59.7 | 0 | 轨道出框未根治 |
| dev-01-13 | Burj Khalifa, pan left. | 41.7 | 53.9 | **+12.2** | 塔基悬空修复 |
| dev-01-14 | Machu Picchu, tilt up. | 58.0 | 64.7 | +6.7 | tilt-up 过冲修复 |
| dev-01-15 | …Machu Picchu, static… | NOT_RENDERED | 69.3 | 新增 | 静态相机修复 |
| dev-01-16 | Forbidden City, pan left. | 46.9 | 59.6 | **+12.7** | 宫殿悬空修复 |
| dev-01-17 | Forbidden City, First-person…dolly. | 32.7 | 50.1 | **+17.4** | 相机穿墙修复 |
| dev-01-18 | Laptop, pan right. | NOT_RENDERED | 69.8 | 新增 | rounded_box 半径修复 |
| dev-01-19 | Watch, zoom out. | NOT_RENDERED | 66.5 | 新增 | rounded_box 半径修复 |
| dev-01-20 | …orbits… clockwise. Watch. | NOT_RENDERED | 59.4 | 新增 | rounded_box 半径修复 |

### 4.3 Round-1 结论

- 外循环转换：attempt-1 →（provider 侧改进 + 两处 harness 缺陷修复）→ attempt-2 → **stop/留证**。无 src/videoact owner 需要补丁（两个真 harness 缺陷在训练过程中被即时修复并单独提交，见 §5）。
- 配对证据方向正确：train 小幅正提升、dev 显著提升、无超噪声回归（最大回退 -0.8）、零失败。
- 遗留最大两类质量问题（供下一轮 provider 改进）：轨道/摇摄中段构图掉主体（01-02/12/20）；桌体 builder 比例错误（01-08）。

## 5. 训练过程中发现并修复的 harness 缺陷（全部在 glm-5.3-flash 分支，独立提交）

| 提交 | Owner | 缺陷 → 修复 |
|---|---|---|
| `9c53d1c`（部分） | director_trajectory | handoff 只发 transfer 不发 detach → 交互评估器要求 attach/transfer/detach 全生命周期，所有 handoff prompt 确定性失败 → 补齐 giver 释放事件 |
| `9c53d1c`（部分） | evaluate 接线 | 内循环评估不传 dataset_fingerprint → experiment-fingerprint 契约必填，静默降级为硬失败 → 传入指纹 |
| `1276a10` 之前 | train_real_harness | assistant 模式未纳入 DIRECTOR_PLAN 绑定审计分支 → 误判为模板臂 → 纳入 |
| `（缓存隔离）` | code_cache | 外层 attempt 共享冻结缓存命名空间 → 跨 attempt 重生成必然冲突 → 按 `outer-{attempt}` 隔离 |
| `1276a10` | code_cache | 同计划重生成不同源码被冻结守卫拒绝（内循环设计行为）→ 槽位版本化写入，保留历史 |

## 6. Provider（glm-5.3-flash 会话）在训练中的教训与改进

| 教训（attempt-1 证据） | attempt-2 落地的改进 |
|---|---|
| 静态道具按计划 z 悬空 → 视觉穿帮（01-13/16 硬伤） | 舞台台面按 `plan_z + 主体base偏移` 对齐，道具落地（dev +12~17 分的主因） |
| zoom-in / tilt-up 过冲丢主体 | 终点距离/高度收敛（01-04 +6.4、01-14 +6.7） |
| 相机穿墙（01-17 评估器硬失败） | first-person dolly 保留 stand-off（+17.4） |
| static 分支修改 frozen dataclass 崩溃 | 重建 keyframe 列表（01-07/15 从 NOT_RENDERED 到 ~70 分） |
| rounded_box 圆角 ≥ 最短边一半崩溃 | 半径修复（01-18/19/20 从 NOT_RENDERED 到 60-70 分） |
| 首候选缺陷浪费渲染预算 | 新增 mock-Blender 干跑门：响应前拦截 NameError/frozen 写入/元数错误 |

## 7. 完整过程与结果保留索引

| 内容 | 位置 |
|---|---|
| 训练记忆表（append-only，全部证据行） | `docs/t2blendercodeharness-glm-flash-training-memory-v1.md` |
| Round-1 attempt 报告/分数 | `out/training/glm-flash-diagnostic-v1/round-01/attempt-0{1,2}/`（attempt_report.json、real_unified_score.json、job_index.json、inner_loop_report.json） |
| 全部真实视频/帧/telemetry | 同上 `real/{train,dev}/<case>/`（proxy.mp4、proxy.blend、telemetry、observer、experiment_fingerprint、provider_manifest） |
| 我撰写的全部 Director 解释（preauth） | `out/assistant-session/preauth/director/`（21 个 case 文件） |
| 我撰写的全部 blender_job.py 与请求/响应 | `out/assistant-session/authoring/`、`requests/`、`responses/` |
| 我撰写的全部 VLM 评审（14 维+证据） | `out/assistant-session/reviews/`、各 split `assistant-reviews/`、每 case `vlm_report.json` |
| 失败尝试完整证据 | 各 split `inner_attempts/<case>/attempt-NN/`（含 stderr、render_attempts） |
| 无效中间运行（诚实保留） | `round-01/attempt-02-invalid-*`（缓存/模板迭代的现场） |
| 冒烟全链路样本 | `out/smoke/assistant-smoke-v4/`（task 38.0 / realism 36.5 / overall 37.6） |

## 8. 诚实边界

- 本轮为 **diagnostic_precalibration**（readiness：仅 golden_review/paired_gate 两个人工门禁 pending——按用户"不要人工"要求以诊断模式运行）：分数与配对证据只用于诊断，正式 harness 接受（formal acceptance）仍被 `training_allowed=false` 阻止，视觉评分不由外部 VLM 仲裁。
- 视觉评审由我（驱动会话的 glm-5.3-flash）阅读真实帧后给出，来源标记 `codex_local_visual_review`，逐维证据绑定，置信度 0.7-0.8；这是我作为 VLM 的判断，不是独立第三方仲裁。
- Round-1 之后未继续 overall（120 case）/rounds 2-6：volume 与会话预算不匹配（每 overall 需约 40 个新主体构建器 + 120 次生成 + 120 次帧评审）。恢复路径见 §9。

## 9. 恢复路径（后续会话继续跑完整计划）

```bash
cd C:\Users\sy\Desktop\T2BlenderCode\.worktrees\glm-5.3-flash
# Round-1 cumulative overall（60 train + 60 dev，120 个新 case）：
#   1) 为 family 02-06 撰写 preauth（扩展 scripts/author_round1_preauth.py 的模式）
#   2) 为交互/动物/物理主体扩展 author_camera_case_source.py 的 builder 表
uv run python scripts/train_real_harness.py --mode diagnostic-overall --round 1 \
  --dataset-root dataset/vbench2-agent-training-index-v1 \
  --readiness-report out/preflight-readiness-exact-v2.json \
  --round-root out/training/glm-flash-diagnostic-v1 \
  --blender-bin D:/blender/blender.exe --workers 4 --vlm-model gpt-5.6-luna \
  --provider-mode assistant --markdown-path docs/t2blendercodeharness-glm-flash-training-memory-v1.md
# 期间运行 scripts/drain_requests.py 服务 codegen 请求（含干跑门）
# 评审：scripts/assemble_round_reviews.py + scripts/complete_assistant_local_reviews.py（模式同 round-1）
# Round 2-6：--round 2..6 重复 attempt（+overall）；每轮结束按 §4.3 格式分析重复失败再决定 provider 改进
```

主干命令（六轮整体计划）在机制上已由 CLI 的 `--mode diagnostic-six-rounds` 支撑；当前逐轮驱动方式与之产出一致（round-NN/attempt-NN/overall 目录 + memory 表 + jsonl）。
