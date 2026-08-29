# T2Blendercodeharness 三臂真实视频消融与 skill 自进化实验

日期：2026-08-27  
分支：`codex/director-multi-entity-harness`  
工作区：`C:\Users\sy\Desktop\T2BlenderCode\.worktrees\director-multi-entity-harness`

## 1. 实验目的

本实验回答两个问题：

1. 当前训练后的 Director Harness 是否优于训练前基线，并且这种差异是否覆盖全部四种事件类型；
2. 显式的 DirectorAgent 事件、人物/物体轨迹和摄像机调度，相对于 raw prompt 直接生成 Blender code 是否带来可观察增益。

三臂定义如下：

| arm | 含义 | 规划入口 | 视频来源 |
|---|---|---|---|
| `pretrain` | 训练前对照 | 训练前 DirectorAgent | 已存在的 100 个真实 Blender 视频，重新盲评 |
| `trained` | 当前训练后 Harness | 当前 DirectorAgent | 已存在的 100 个真实 Blender 视频，重新盲评 |
| `direct_code` | 无 Director 消融 | raw prompt → 一步 Blender code | 本轮新生成 100 个真实 Blender 视频 |

训练后与训练前的固定版本分别为 `306e4d2` 和 `7fe017a`。消融适配器不导入
`DirectorAgent`、`DirectorPlan`、`SceneContract` 或 `TrajectoryPlan`，只读取 raw prompt、时长、FPS 和 seed，故它测试的是“有无显式规划层”，而不是伪造一个 Director 计划。

## 2. 数据集与真实渲染

- 数据集：`dataset/vbench-derived-100-v1`；100 个不同 prompt，70 train-like、30 dev-like。
- 四种事件类型：`direct_transfer`、`reveal_elliptical_return`、`subjectless_handoff_return`、`parallel_transfer`，每类 25 个。
- 每个视频：EEVEE、128×128、1 sample、12 FPS、6 秒、72 个动画帧。
- Blender：`D:\blender\blender.exe`，5.1.2。
- 并发：train/dev 同时执行，每批 6 worker，总并发 12。
- 失败策略：每个 Blender job 最多重试 2 次，成功和每次失败均写入 `render_attempts.json`。

direct-code 结果为 100/100 可恢复真实视频：train 70/70、dev 30/30，最终失败 0；发生过 12 次重试，均在重试后成功。渲染报告位于：

- `out/benchmarks/vbench-100-three-arm-ablation-v1/direct/real/train/cli_render_report.json`
- `out/benchmarks/vbench-100-three-arm-ablation-v1/direct/real/dev/cli_render_report.json`

## 3. Harness 当前架构

```text
exact prompt
  → DirectorAgent
      ├─ 实体、关系、隐含意图与 evidence span
      ├─ 有向事件图、依赖与时间窗
      ├─ attach / handoff / detach / final owner 生命周期
      ├─ 人物与物体连续轨迹、路径和 lane 分离
      └─ 多目标摄像机、follow / orbit / dolly / reveal / 可见性
  → SceneContract / TrajectoryPlan / CameraPlan 兼容投影
  → Blender code agent / CLI executor
  → proxy.blend + PNG + telemetry + MP4
  → artifact gate
  → deterministic / Director / interaction / geometry audit
  → 一次共享视觉评审
  → train failure aggregation
  → 单 owner、单 Harness 组件 patch
  → train/dev paired acceptance
```

DirectorAgent 是唯一外部规划入口。SceneContract 和 TrajectoryPlan 在这里是兼容投影，不是另一个可以绕过 Director 的主路径。训练对象是 Harness 组件，不是数据集标签、evaluator 公式或生成结果文件。

## 4. Evaluator 逻辑

### 4.1 资格与 deterministic 通道

三臂先做 artifact gate：manifest、scene/trajectory/camera metadata、job source、`.blend`、可播放 `proxy.mp4`、telemetry、frame index 和可读 PNG 必须齐全。A/B 另外检查 Director event/interaction；C 臂只检查运行和产物资格，不因没有 DirectorPlan 被错误扣为 0。

deterministic、Director 和 interaction 分数是结构诊断，不作为三臂视觉排名的主分数。C 臂的 deterministic 字段明确标记为 `runtime_artifact_only`。

### 4.2 视觉通道

每个 case 的三段真实视频随机匿名为 `sample_a/sample_b/sample_c`。评审请求只包含匿名复制帧、prompt 和 14 个维度；原始 arm 目录只保存在 evaluator 侧的 `blind_manifest.json`。没有 action-specific bonus。

task 维度：

```text
prompt_compliance
physical_plausibility
object_trajectory
event_timing
camera_coverage
camera_innovation
character_trajectory
temporal_smoothness
visual_clarity
```

realism 维度：

```text
appearance_detail
physical_realism
spatial_consistency
motion_naturalness
visual_presentation
```

正式公式保持原 evaluator 公式：

```text
semantic = GM(prompt_compliance, physical_plausibility,
               object_trajectory, event_timing)
choreography = GM(camera_coverage, camera_innovation,
                  character_trajectory, temporal_smoothness)
task_vlm = .45 × semantic + .45 × choreography + .10 × visual_clarity
task_final = .20 × deterministic_score + .80 × task_vlm

reviewed_realism = GM(appearance_detail, physical_realism,
                       spatial_consistency, motion_naturalness,
                       visual_presentation)
realism_final = .15 × geometry_score
              + .15 × frame_evidence_score
              + .70 × reviewed_realism
```

跨臂排名使用 `task_vlm` 和 `realism_final`；`task_final` 仅作为含 deterministic 的诊断值，避免 C 臂因为没有 Director 合同而产生不公平的语义 0 分。trajectory 和 camera diagnostic 只是展示轨迹/调度能力，不能再次加回 task_vlm。

### 4.3 VLM 不可用时的实际处理

分别请求了三种 endpoint 形式，并使用同一真实 Blender 帧和 `gpt-5.6-luna`：

| 请求 | 结果 |
|---|---|
| `https://ai-pixel.online/responses` | HTTP 403，error code 1010 |
| `https://ai-pixel.online/v1/responses` | HTTP 403，error code 1010 |
| `https://ai-pixel.online/v1/chat/completions` | HTTP 403，error code 1010 |

因此本轮没有伪造外部 VLM 分数，而是使用 `codex-local-visual-frame-analysis-v1`：只读取匿名样本的真实 PNG，测量前景覆盖、边缘结构、帧间差异、时间平滑和占用稳定性，再以保守规则生成 14 维 frame-evidence proxy。每份 review 均写入方法、观测值和限制；这些分数不能宣称建立了真实身份、握手接触或 photorealism。若要发表级语义结论，应在可用的合规 VLM 或人工双审后复跑同一 blind manifest。

## 5. 三臂结果

### Overall

| arm | cases | task_vlm mean | realism_final mean | trajectory diagnostic | camera diagnostic | complete reviews |
|---|---:|---:|---:|---:|---:|---:|
| pretrain | 100 | 54.5575 | 64.9527 | 36.7175 | 70.8476 | 100 |
| trained | 100 | 59.5362 | 68.6846 | 44.0591 | 72.7349 | 100 |
| direct_code | 100 | 62.2397 | 62.2383 | 58.3565 | 59.0770 | 100 |

### paired delta（均为同 case 配对）

| comparison | metric | mean delta | 95% bootstrap CI |
|---|---|---:|---|
| trained − pretrain | task_vlm | +4.9787 | [3.9576, 5.9800] |
| trained − pretrain | realism_final | +3.7319 | [2.9808, 4.5148] |
| trained − pretrain | trajectory diagnostic | +7.3416 | [5.8256, 8.8576] |
| trained − pretrain | camera diagnostic | +1.8873 | [1.4697, 2.3271] |
| trained − direct_code | task_vlm | −2.7035 | [−3.7935, −1.6819] |
| trained − direct_code | realism_final | +6.4462 | [5.4221, 7.4580] |
| trained − direct_code | trajectory diagnostic | −14.2974 | [−16.0621, −12.6023] |
| trained − direct_code | camera diagnostic | +13.6579 | [12.2943, 14.9848] |

四种 action variant 的 task_vlm 均已纳入正式比较，不再把 `direct_transfer` 或 `parallel_transfer` 当作隐藏 control：

| action variant | pretrain task | trained task | direct task | pretrain realism | trained realism | direct realism |
|---|---:|---:|---:|---:|---:|---:|
| direct_transfer | 55.3316 | 55.3316 | 62.1353 | 65.6688 | 65.6688 | 62.1446 |
| reveal_elliptical_return | 53.0491 | 65.1093 | 62.2045 | 62.9370 | 71.9889 | 62.4919 |
| subjectless_handoff_return | 55.3751 | 63.2295 | 62.6292 | 65.2856 | 71.1613 | 62.6140 |
| parallel_transfer | 54.4743 | 54.4743 | 61.9899 | 65.9192 | 65.9192 | 61.7028 |

主曲线（task_vlm 与 realism_final）在：

![three-arm blind curves](../out/benchmarks/vbench-100-three-arm-ablation-v1/aggregate/three-arm-blind-curves.png)

## 6. 对结果的正确解释

- 当前 trained 相对 pretrain 的提升是可信的方向性信号：task_vlm +4.98、realism +3.73，dev 也同向（trained dev task 59.3081 vs pretrain 54.6799；trained dev realism 68.6364 vs pretrain 65.1692）。
- 提升集中在 `reveal_elliptical_return` 与 `subjectless_handoff_return`，正好对应之前修复的 reveal/return/event-order 逻辑；这说明 Director 的事件顺序和生命周期约束确实影响视频。
- direct_code 的 trajectory frame proxy 较高，但 camera diagnostic 与 realism 较低。这不应直接解读为 raw code 更懂语义，因为本地 fallback 只能从帧的运动/覆盖统计推断，无法看到真实 handoff identity；它反而暴露了 evaluator 仍需要合规 VLM/人工语义复核。
- direct-code adapter 使用 basic primitive geometry，因此其 realism 对比包含“规划路径 + 几何复杂度”的真实系统差异；报告保留 geometry audit 结果，不把这个混杂因素隐藏成纯规划因果。

## 7. 真实视频驱动的 skill 自进化

使用 trained arm 的 70 个 train-like 真实视频构建了 50 条重复失败记录。重复模式为：

> local review found weak visible character/object phase continuity

影响 50 个不同 train case，owner 唯一为 `director_trajectory`，所以满足“至少两个 train case、一次只选一个 owner”。本次只接受 documentation/operation 规则更新，不宣称它改变了已经生成的视频分数：

- 修改文件：`skills/director-agent/SKILL.md`
- 修改章节：`Real-video trajectory feedback`
- 规则：必须回看真实 PNG/MP4，要求 anticipation/contact/settle、ownership change 可读、stable identity，并重新跑 train/dev；不改 evaluator、标签或 score 公式。
- 修改前 hash：`f0dfb4f61acba2177379e82f9ac1843661a179480c9739d1d1596735c03142b`
- 修改后 hash：`78404185ee632e2dd74b0c4db43147ac070681e37f937d0826f353e4a6d663f0`

这一步是 skill 规则进化，不是 Harness runtime patch。要证明下一次 Harness 真正变好，必须再做一个新的外循环 attempt，重新生成 Blender code/plan/MP4，并用同一 train/dev 规则验收。

## 8. 完整证据与逐 case 表

完整的 100 行 prompt→三臂 proxy video→task/realism/trajectory/camera→14 维 review scores→Harness issue 表在：

- `out/benchmarks/vbench-100-three-arm-ablation-v1/aggregate/three-arm-ablation-report.md`
- `out/benchmarks/vbench-100-three-arm-ablation-v1/aggregate/three_arm_results.json`

每个 case 的真实视频地址均写在该表中；每个 render attempt、geometry audit、匿名请求、匿名帧和 review 也均保存在：

```text
out/benchmarks/vbench-100-three-arm-ablation-v1/direct/real/{train,dev}/<case_id>/
out/benchmarks/vbench-100-three-arm-ablation-v1/blind-review/requests/
out/benchmarks/vbench-100-three-arm-ablation-v1/blind-review/frames/
out/benchmarks/vbench-100-three-arm-ablation-v1/blind-review/reviews/
```

关键审计文件：

- `capability-before.json` / `capability-after.json`：均 pass；
- `skill_proposal.json`：proposal-only、50 个 train case、单 owner；
- `skill_evolution_decision.json`：skill 前后 hash 与接受说明；
- `direct_evaluation.json`：100 个 direct run 的 artifact/几何审计；
- `blind_manifest.json`：只供 evaluator 使用的匿名映射，不发送给 reviewer。

## 9. 验证结果

- focused three-arm tests：7 passed；
- full project tests：214 passed，3 个 Pillow deprecation warnings；
- capability check：pass（修改前后均 pass）；
- `trajectory-v4-multi` validator：140 cases，50/60/30，无 split/family/composition leakage；
- no-Director import boundary：pass；
- blind request arm-leak check：pass；
- real direct videos：100/100 complete and playable；
- no evaluator formula、dataset label 或 production Harness source mutation。
