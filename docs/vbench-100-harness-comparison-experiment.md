# VBench-derived 100 Harness 对比实验记录

## 结论先行

在同一批 100 条 VBench-2.0-derived prompt 上，训练后的 current
Harness 相对训练前 baseline 的 task 平均分提高 **8.784**，dev 子集提高
**8.783**；realism 平均分提高 **1.206**。current 在 50 个事件链样本上
胜出，另外 50 个 direct/parallel 控制样本持平，没有 paired loss。

这个提升不能直接表述为“全面泛化”：50 个胜出样本有意覆盖上一轮训练
修复的 reveal/elliptical-return 与 subjectless-handoff-return 事件链，而
另 50 个控制样本没有变化。它验证了修复在新的 VBench 场景语境中仍然能
触发，但没有把定向回归结果冒充成所有 prompt 类型的泛化结论。

## 数据来源与构建

来源是 VBench-2.0 官方 prompt 资产
[`VBench2_full_info.json`](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/VBench2_full_info.json)，
方法背景参见官方 CVPR 2024 VBench 论文
[`VBench`](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf)。

本实验不是直接声称复现 VBench 原始 benchmark 分数，而是构建
`VBench-derived-100`：

- `Camera_Motion`、`Human_Interaction`、`Motion_Order_Understanding`、
  `Complex_Plot`、`Dynamic_Spatial_Relationship` 五类各抽取 20 条；
- 每类 14 条 train-like、6 条 dev-like，共 70/30；dev 从未用于修改
  Harness；
- 原始 `source_prompt`、source dimension、原始 JSON index、auxiliary cue
  和 source SHA-256 全部保留；
- 可执行 `prompt` 保留 VBench seed 作为场景上下文，再补充两个 named
  actor、两个 proxy object、事件顺序、camera coverage 和负约束；这一步
  是为了让当前 Blender Harness 的轨迹规划和摄像机调度有可观察的执行
  契约；
- 所有 case prompt 唯一；所有视频由同一套代理几何生成，不能把代理物体
  的外观误报为真实 VBench 场景外观。

数据集位置：`dataset/vbench-derived-100-v1`。本地源文件位置：
`data/vbench-source/VBench2_full_info.json`。

## 比较版本与执行设置

| 项目 | baseline | current |
|---|---|---|
| Harness 代码 | `7fe017a`，训练前 | `306e4d2`，完成多轮 Director 训练后 |
| label | `h-t2-hard-v4-pretraining-baseline` | `h-t2-hard-v4-director-prompt-elliptical-return-order-v1` |
| Blender | `D:\blender\blender.exe`，5.1.2 | 同左 |
| renderer | EEVEE Next | 同左 |
| resolution/samples | 128×128 / 1 | 同左 |
| fps/duration | 12 fps / 6 s，72 帧 | 同左 |
| evaluator | `real-v4-shared-evidence-separate-scores` | 同左 |
| VLM | 外部 endpoint 未使用 | 外部 endpoint 未使用 |
| visual review | Codex local，exact 8 sampled frames | 同左 |

每个 case 都真实生成 `proxy.blend`、72 张 animation PNG、telemetry、MP4
和 render attempt log。四个 split 共 200 个视频：baseline/current ×
train/dev。Blender CLI 使用 12 个并发 worker（四个 split 每个 3 个），
每个 job 最多 2 次自动重试；最终 200/200 成功且 0 次重试。

## 评测逻辑

### Task

deterministic/director/interaction 负责结构和硬门检查，但不作为主要视觉
结果。通过 artifact gate 后，用 8 个 chronological frame 做同一份 local
review。task review 先对下列四个 semantic 维度求几何平均，再对四个
choreography 维度求几何平均：

```text
semantic = GM(prompt_compliance, physical_plausibility,
               object_trajectory, event_timing)
choreography = GM(camera_coverage, camera_innovation,
                   character_trajectory, temporal_smoothness)
task_vlm = 0.45 * semantic + 0.45 * choreography + 0.10 * visual_clarity
task_final = 0.20 * deterministic + 0.80 * task_vlm
```

这里的 `task_final` 是 task channel；它没有与 realism 相加。

### Realism

realism 独立于 task channel。Blender geometry audit 与 PNG evidence 先作
artifact safeguard；local review 的五个真实性维度为主信号：

```text
reviewed_realism = GM(appearance_detail, physical_realism,
                       spatial_consistency, motion_naturalness,
                       visual_presentation)
realism_final = 0.15 * geometry_score
              + 0.15 * rendered_frame_evidence
              + 0.70 * reviewed_realism
```

本批所有 review 都明确记录了白膜、低分辨率、低面代理几何和后段 close
framing 限制，因此 realism 没有被写成 photorealism 结论。几何 hard gate
为 0 只说明代理 artifact 完整、几何审计通过，不说明画面真实。

## 结果

| 集合 | variant | case 数 | task mean | realism mean | deterministic mean |
|---|---|---:|---:|---:|---:|
| all | pretrain | 100 | 62.8783 | 45.8103 | 100.0000 |
| all | current | 100 | 71.6622 | 47.0160 | 100.0000 |
| train-like | pretrain | 70 | 62.9064 | 45.7921 | 100.0000 |
| train-like | current | 70 | 71.6909 | 47.0266 | 100.0000 |
| dev | pretrain | 30 | 62.8127 | 45.8527 | 100.0000 |
| dev | current | 30 | 71.5954 | 46.9914 | 100.0000 |

### 事件链诊断

| action pattern | pretrain task | current task | task delta | pretrain realism | current realism |
|---|---:|---:|---:|---:|---:|
| direct_transfer | 71.0249 | 71.0249 | 0.0000 | 46.9753 | 46.9753 |
| reveal_elliptical_return | 55.1250 | 72.7006 | +17.5756 | 44.3908 | 47.2046 |
| subjectless_handoff_return | 56.5815 | 74.1418 | +17.5603 | 45.5656 | 47.5748 |
| parallel_transfer | 68.7816 | 68.7816 | 0.0000 | 46.3094 | 46.3094 |

## 本地审查与产物

- 完整 paired 结果、每个 prompt、两个 proxy 视频绝对路径、两边分数、
  delta、finding 解释：
  `out/benchmarks/vbench-100-current-vs-pretrain-v1/summary/vbench-100-current-vs-pretrain-report.md`；
- 机器可读 paired JSON：
  `out/benchmarks/vbench-100-current-vs-pretrain-v1/summary/paired_results.json`；
- task/realism 曲线：
  `out/benchmarks/vbench-100-current-vs-pretrain-v1/summary/vbench-100-current-vs-pretrain-curves.png`；
- 20 张 exact sampled-frame contact sheets：
  `out/benchmarks/vbench-100-current-vs-pretrain-v1/review-sheets`；
- Codex local review JSON：
  `out/benchmarks/vbench-100-current-vs-pretrain-v1/local-reviews`；
- 四个 split 的 render success/retry 报告位于各自
  `real/<split>/cli_render_report.json`；
- 原始 baseline/current job、blend、PNG、MP4、telemetry 及 evaluator 报告
  位于：
  `out/benchmarks/vbench-100-current-vs-pretrain-v1/{pretrain,current}/real/{train,dev}`。

本实验只新增 benchmark 构建/汇总脚本和文档，没有因本轮比较修改
`src/videoact` Harness；后续若要继续进化，应把这 100 条分成新的训练集
与真正冻结的第三方式测试集，避免继续在同一批 VBench-derived prompt 上
调参。
