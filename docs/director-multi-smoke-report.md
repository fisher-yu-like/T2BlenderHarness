# DirectorAgent 多实体真实 Smoke 报告

日期：2026-08-26  
分支：`codex/director-multi-entity-harness`  
Harness：`t2blendercodeharness-v3` / Director policy `director-v1`

## 结论

真实 Blender CLI smoke 已通过 artifact、确定性任务、Director、interaction、几何和可播放视频 gate。测试使用本机 `D:\blender\blender.exe`（Blender 5.1.2），不是伪造视频或伪造分数。

外部 VLM 没有可用 endpoint，因此没有生成 VLM 分数，也没有把 unavailable 转成 0 或满分。当前 realism 分数是独立的 artifact-only proxy 估计，明确标注为不能代表完整视觉真实性。

## 固定输入

| 项目 | 值 |
|---|---|
| 数据集 | `dataset/trajectory-v4-multi` |
| 数据集 fingerprint | `4d0abd0eb387bc58c4b4cd03259874670a5c7010ecbc484f056cbf3b255a0f41` |
| 样本 | `multi-train-001` |
| Prompt | Alice 携带 green ball，交给 Carla，Carla 放置；随后 Carla 通过独立轨道处理 yellow book；要求广角建立、曲线跟随、短 rack-like reveal、左侧入场和可读交接。 |
| 渲染目录 | `out/smoke/director-multi-v3/multi-train-001` |
| Blender | `D:\blender\blender.exe`, 5.1.2 |
| 分辨率 / fps / 时长 | 256×256 / 24 / 20 秒 |

## Gate 结果

| 通道 | 结果 | 说明 |
|---|---:|---|
| Blender CLI | PASS | return code 0；480 个 animation frames |
| 重试 | 0 | 第一次即完成；最多允许 2 次重试 |
| MP4 | PASS | `proxy.mp4` 可播放，20 秒，24 fps |
| artifact gate | PASS | `proxy.blend`、telemetry、frame index、Director hash 均完整 |
| deterministic task score | 100.0 | 不含 Director 分数，不含 realism 分数 |
| Director plan score | 100.0 | 无 evidence、依赖、身份、轨迹碰撞或 camera coverage finding |
| interaction | PASS | giver/receiver、attach/transfer/detach、handoff window 和 final support 均成立 |
| geometry score | 100.0 | 6 个实体均有非 primitive 的 detailed parametric mesh；hard fail 0 |
| frame evidence score | 68.9528 | 3 个可读采样帧；只表示帧结构可用 |
| artifact-only realism | 68.5519 | 独立通道；`realism_claim=not_established`，上限 80 |
| VLM | unavailable | 未调用外部 endpoint，未合成分数 |

关键产物：

- 视频：[proxy.mp4](../out/smoke/director-multi-v3/multi-train-001/proxy.mp4)
- 确定性报告：[deterministic_report.json](../out/smoke/director-multi-v3/multi-train-001/deterministic_report.json)
- Director 计划：[director_plan.json](../out/smoke/director-multi-v3/multi-train-001/director_plan.json)
- 轨迹与交互 telemetry：[telemetry.json](../out/smoke/director-multi-v3/multi-train-001/telemetry.json)
- 重试记录：[render_attempts.json](../out/smoke/director-multi-v3/multi-train-001/render_attempts.json)
- 几何报告：[geometry_report.json](../out/smoke/director-multi-v3/multi-train-001/geometry_report.json)
- realism 报告：[realism_report.json](../out/smoke/director-multi-v3/multi-train-001/realism_report.json)
- 交接中点帧：[frame_0241.png](../out/smoke/director-multi-v3/multi-train-001/frames/animation/frame_0241.png)

## 发现与修复记录

第一次真实渲染证明 Blender CLI、proxy.blend、telemetry、MP4 和重试机制可以跑通，但原始 prompt 的动作措辞使 Director interpreter 只解析出单个 carry。数据集 prompt 随后改成显式的 `carries → hands ... to ... → places` 事件顺序；这属于数据集构造修正，不是通过改标签刷分。

第二次 smoke 的 evaluator 又把合法交接终点误报成 `director_path_collision`。根因是 `13.3333 × 24` 的浮点误差导致 evaluator 得到 320 帧，而轨迹 composer 按四舍五入使用 321 帧。已在 `evaluator/director_metrics.py` 统一使用 `round(seconds × fps) + 1`，并加入回归测试。修复后第三次真实渲染通过，报告中的 Director 100.0 是修复后的独立结果。

## 人工可视检查

检查帧为交接窗口中点附近的 `frame_0241.png`，并对比相邻帧。确定性 telemetry 明确记录了 Alice→Carla 的 transfer（frame 161）、Carla 最终 owner 和 support contact；但灰膜 256×256 画面中两个角色在交接区域有重叠，绿色球的材质/颜色辨识度不足。因此本 smoke 只宣称“轨迹与交互逻辑通过”，不宣称“视觉真实性已通过”。这一项会作为训练期 realism baseline，不会修改 evaluator 以抬高分数。

## 复现与验证

本阶段最后一次验证结果：

```text
198 passed in 6.73s
capability_check: pass
validate_multi_entity_dataset: pass
train/dev/test: 50/60/30
family_overlap: []
composition_overlap: []
```

下一阶段按照训练 Skill 进入五轮外循环：每轮至多 5 次尝试，每次 10 个 train + 10 个 paired dev，每轮结束评估已见 train 与完整 60 个 dev；只允许一次一个 Harness owner 的 patch，Director/task/realism 分数始终分开记录。
