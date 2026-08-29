# Active failure sampling replay protocol

版本：`active-sampling-v2-replay`

T16 只允许用已经完成的 train/dev 历史记录审计预算控制。它不是新的训练数据，也不能读取 frozen/test 结果。每个 replay batch 必须包含：

```json
{
  "batch_id": "round-01-screen-01",
  "records": [
    {
      "case_id": "train-case-001",
      "split": "train",
      "paired_delta": 2.4,
      "render_cost": 4,
      "review_confidence": 0.72,
      "judge_disagreement": 0.20,
      "findings": [
        {"root_cause_id": "camera:coverage", "owner": "camera_planner", "severity": "error"}
      ]
    }
  ],
  "full_decision": "stop_success"
}
```

`paired_delta` 是该 case 的预先记录的 paired 主指标差值；`render_cost` 是该 case 的真实渲染/重试成本，缺省为 1。`full_decision` 可选，但如果提供，必须与用全部 delta 重新计算的 decision 一致。采样决策始终只用 sampler 选出的 case 重新计算，不能由人工填入。

运行：

```powershell
uv run python scripts/audit_active_sampling.py `
  --replay path/to/historical-replay.json `
  --budget 10 `
  --target-lower-bound 1.0 `
  --min-cases 10 `
  --min-reduction 0.30 `
  --min-agreement 0.95 `
  --out out/preflight/active-sampling-replay.json
```

通过条件固定为：总 render cost 至少下降 30%，且采样与全量的 stop/continue 决策一致率至少 95%。缺少 `paired_delta`、使用 test split、重复 case ID、非数值成本或历史决策与重算不一致时直接失败。报告仅保留 case ID、选择原因、概率和聚合指标，不复制 prompt、test score 或原始视频内容。

当前仓库只有 API 和回放测试；没有把旧实验结果强行转换成历史 replay，也没有把合成 fixture 当作正式证据。正式 G3 仍需真实历史记录或一次不改 Harness 的 shadow round。
