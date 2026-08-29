# Realism evaluator v3 审计说明

> **Historical snapshot.** The active realism boundary is
> [evaluator-v5-calibration.md](evaluator-v5-calibration.md): geometry and
> frame statistics are artifact evidence, while an eligible visual review is
> required for a realism claim.

更新时间：2026-08-26

## 结论

旧版 `realism-v2-independent-geometry` 的 100 分不能解释为视频真实感。它检查的是 renderer 自己声明的结构条件：实体都存在、类型一致、顶点/面数超过阈值、没有命中 primitive 名称、带有 `detailed_parametric_v1` 标记、连通组件数达到下限。candidate-v2 恰好全部满足这些条件，所以 100 分是“合同合规饱和”，不是独立真实感。

v3 将 geometry audit 降级为 eligibility evidence，增加真实 PNG 帧的低层测量，并为未完成独立视觉审查的结果保留 20% 未观测质量空间。旧报告 `out/complex-realism-v1/evolution-v1/realism_evolution_v2_report.json` 只作为历史记录，已被 v3 报告 supersede。

## 三层证据

1. `geometry_compliance`：Blender 内打开真实 `proxy.blend`，检查 required entity coverage、entity kind、topology、primitive hint、geometry style 和 connected components。hard failure 会阻断正常 realism claim，并将几何质量限制到 20。
2. `rendered_frame_evidence`：读取 `frames/index.json` 指向的真实 PNG，检查可读性、分辨率、相对背景的前景覆盖率、边缘结构和相邻采样帧的像素变化。它不从 plan 或 telemetry 推断“这是人”或“发生了 grasp”。
3. `independent_review`：只有真实、完整、置信度至少 0.6 的 `gpt-5.6-luna`、`gpt-5.6-terra` 或人工 review 才能确认语义、动作、摄像机和 presentation。当前 endpoint 不可用时不补造该层。

## 公式

几何质量使用连续指标：

```text
G = .20 coverage
  + .20 topology_detail
  + .20 non_primitive_representation
  + .10 semantic_integrity
  + .30 structural_detail
```

`topology_detail` 按顶点/面数相对目标的对数比例计算；`structural_detail` 按 entity kind 的组件目标和 detailed style 计算。因此“刚过 256 vertices”与“真正有更高拓扑”的 case 不再同分。

真实帧分 `V` 使用五项权重：availability `.20`、resolution `.15`、foreground coverage `.25`、edge structure `.20`、temporal change `.20`。

无独立 review 时：

```text
A_raw = .60 × G + .40 × V
artifact_only_score = min(80, .80 × A_raw)
```

`.80` 代表低层 artifact 指标无法观测的语义、物理、轨迹可见性、交互接触、遮挡正确性和动作质量。若低层指标全部为 100，仍然最多 80，且报告中 `realism_claim=not_established`。

独立 review 完整后才使用：

```text
R = .45 semantic + .45 choreography + .10 presentation
final_review_fused = .20 artifact_only_score + .80 R
```

其中 `semantic` 覆盖 prompt 事件顺序、人物/物体身份、抓取/携带/放置和物理关系；`choreography` 覆盖 follow/orbit/dolly、人物轨迹、物体轨迹、事件可见性、时间连续性；`presentation` 覆盖构图、遮挡、清晰度、光照与可读性。各项必须保留可回看的帧证据。

## 本次真实重算

命令实际使用 `D:\blender\blender.exe` 打开两组 `.blend`，并读取两组真实 PNG：

| 对比 | 平均分 | 状态 | 证据 |
|---|---:|---|---|
| baseline（8 cases） | 24.8601 | `artifact_only_weak` | 8/8 geometry hard fail，但视频和采样帧真实存在 |
| candidate-v2（8 cases） | 71.4144 | `artifact_only_proxy`，各 case 69.8615–72.6681 | 8/8 candidate render success，0 geometry hard fail，仍 `not_established` |

candidate 的提升只能表述为“从粗粒度 primitive stand-in 进化到结构化高拓扑 proxy，并且低层画面证据更完整”。不能表述为照片级真实，也不能用它替代人物/物体轨迹和摄像机语义 review。

## 对训练的影响

- 训练记忆中旧 v2 的 100 分标记为 superseded，不删除历史证据。
- 后续每个 case 必须保存 `geometry_report.json`、`visual_evidence.json`、`realism_report.json` 和真实 `proxy.mp4` 路径。
- deterministic/geometry hard failure 不能被 artifact-only 分数抵消。
- 若 `gpt-5.6-luna` 或 `gpt-5.6-terra` 不可用，记录 unavailable；如需语义进入 patch gate，暂停并请求人工 review。
- 真实化 Harness 进化仍遵守单 owner、train 提升、dev 不下降和冻结 test。
