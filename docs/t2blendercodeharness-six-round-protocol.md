# T2Blendercodeharness 六轮真实视频训练协议

这是新的训练协议；此前五轮实验文档保留为历史记录，不与本协议的真实视频分数混用。

## 数据分层

`dataset/trajectory-v3-hard` 现在包含 140 个独立 prompt/proxy scene：

| Split | 数量 | 用途 |
|---|---:|---|
| train | 60 | Harness patch 的失败聚合与训练 |
| dev | 60 | 六轮对应 holdout、累计整体评测、防过拟合门禁 |
| test | 20 | 最终冻结盲测，不参与 patch 选择 |

train 使用 `hard-01` 到 `hard-06`，dev 使用 `hard-07` 到 `hard-12`，每个 family 10 个 case。每个 prompt 和 proxy scene ID 唯一，family 不跨 split 泄漏。

## 六轮安排

| 轮次 | train | 对应 dev | 轮末整体评测 |
|---:|---|---|---|
| 1 | hard-01-01..10 | hard-07-01..10 | 累计 round 1 的 train/dev |
| 2 | hard-02-01..10 | hard-08-01..10 | 累计 round 1–2 的 train/dev |
| 3 | hard-03-01..10 | hard-09-01..10 | 累计 round 1–3 的 train/dev |
| 4 | hard-04-01..10 | hard-10-01..10 | 累计 round 1–4 的 train/dev |
| 5 | hard-05-01..10 | hard-11-01..10 | 累计 round 1–5 的 train/dev |
| 6 | hard-06-01..10 | hard-12-01..10 | 累计全部 60 train/60 dev |

每轮最多执行 5 个外循环 attempt；每次 attempt 用同一 Harness snapshot 生成 10 train + 10 对应 dev，共 20 个真实视频。attempt 结束后停止，聚合 train failures，只允许一个 Harness owner 修改源码，再重新生成 Blender code 和 plan。选出 accepted candidate 后，轮末再用该 candidate 跑完整 60 train + 60 dev，共 120 个整体评测视频。每轮最大 220 个视频，六轮最大 1320 个；不在单个 case 内做 inner-loop 局部修复。

基础设施重试独立于外循环：每个 immutable job 默认最多重试 2 次。Blender 非零退出、超时、缺少基本渲染产物、动画帧不足或 host 组装的 MP4 不可播放时触发；每次记录 `render_attempts.json`，成功重试仍只算该 case 的一个视频，不改变 Harness attempt 计数。

## 防过拟合门禁

```text
train_after > train_before
paired_dev_after >= paired_dev_before
cumulative_overall_dev_after >= cumulative_overall_dev_before
hard_dev_regression == false
```

不满足时记录 `rejected_anti_overfit_gate`，回到 parent Harness。不能改 prompt、case、oracle、evaluator、Blender job、生成 plan 或视频；只可修改 `src/videoact/` 中的 Harness 源码。没有 repeated actionable failure 时记录 `no_patch`。

## 真实评分

Blender CLI 固定为 `D:\blender\blender.exe`，使用并行 worker 渲染独立 case。每个视频必须通过完整 artifact gate，host 从真实 PNG 组装 `proxy.mp4`，再由 deterministic real evaluator 和 VLM 真实采样帧评分。

VLM 只能使用：

- `gpt-5.6-luna`
- `gpt-5.6-terra`

VLM unavailable 不填 0、不使用 fake score，也不进入均分。

若真实 VLM 不可用，改用 `assistant_local_review`：Codex 检查 request 指定的真实时间序列帧，逐项填写九维分数、visible evidence、weaknesses 和 confidence；复核结果必须经过相同 schema/聚合器校验，不能复制 deterministic 分数或由 plan 单独推断。没有本地复核文件的 case 保持 `awaiting_assistant_review`，不进入均分和 patch 选择。

Evaluator 先过 artifact gate（包括实际解码 MP4），再运行声明式 plan/telemetry gate 和结构 oracle。每个 SceneContract 明确声明实体的 required event phases、minimum states、attachment actions；无关 prompt 不套固定 manipulation 状态机。Finding 分为 hard/error/warning/info，并按 root cause 去重：只有 hard 阻断视觉评分；error/warning 只降低 20% 的可靠性分。视觉层使用首尾帧、必看事件中点和均匀补帧组成最多 8 帧，独立评分相机 coverage/innovation、人物/物体轨迹和事件 timing；最终使用 20% deterministic + 80% video review，hard gate 时封顶 49。

结构 oracle 是数据集作者在 runtime plan 之前写入的事件、camera、entity、attachment 期望，用于防止 planner 自证正确；它不是人工视频 ground truth。视觉结果必须提供 evidence 和 confidence，低于 0.6 时暂停并请求人工，不进入均分或 patch 选择。

## 必须保存的 Markdown 记忆表

最终文件：`docs/t2blendercodeharness-six-round-training-memory.md`

每个 case 一行，字段如下：

| 轮数 | Prompt | Proxy 视频地址 | 打分多少 | 评审来源/置信度 | Severity/Root cause | 渲染重试 | 检测出 Harness 什么问题 | 修复 Harness 哪里、怎么修复 | 修复后重新跑提升或下降多少 | 自然语言描述怎么处理 |
|---:|---|---|---:|---|---|---:|---|---|---:|---|

“自然语言描述”必须说明证据、owner、修复决定、接受/拒绝/rollback 原因，以及为什么该结果不能被解释为过拟合后的泛化提升。对应的 JSON 证据保存在每个 `round-XX/overall_evaluation.json`、`patch_manifest.json` 和 `memory/harness_updates.jsonl`。

当视觉置信度低于 0.6、prompt/oracle 有语义歧义或 calibration A/B 无法自动裁定时，立即把视频、关键帧和争议维度呈现给用户。人工只需选择 A/B/持平/无法判断或确认语义，不需要手写 plan、Blender code 或九维分数。

## 执行入口

先审计协议：

```powershell
& "$env:LOCAL_PYTHON" scripts\train_real_harness.py --mode protocol `
  --dataset-root dataset\trajectory-v3-hard `
  --round-root out\training\six-rounds-real-v6
```

批准后执行六轮与最终全量评测：

```powershell
& "$env:LOCAL_PYTHON" scripts\train_real_harness.py --mode all `
  --dataset-root dataset\trajectory-v3-hard `
  --round-root out\training\six-rounds-real-v6 `
  --full-train-root out\training\full-evaluation-real-v6 `
  --blender-bin D:\blender\blender.exe `
  --workers 12 `
  --vlm-model gpt-5.6-luna `
  --markdown-path docs\t2blendercodeharness-six-round-training-memory.md
```
