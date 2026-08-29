# T2Blendercodeharness release bundle v1

这是一个可移交/复现实验的轻量压缩包目录说明。

## 包含内容

- `src/`：Harness 核心 contract、Director、轨迹、摄像机、执行和 meta harness 代码；
- `evaluator/`：deterministic、Director/interaction、geometry、realism、visual-review 和 VLM 接口；
- `blender/`、`scripts/`、`training/`：Blender job、CLI、真实渲染、数据集、评估、memory 和 skill proposal 工具；
- `skills/`：项目全部可复用 skill，包括 `t2blendercodeharness`、中文 skill、`director-agent`、训练和 evolution skill；
- `tests/`、`pyproject.toml`、`uv.lock`：安装、能力检查和回归测试入口；
- `dataset/`、`data/vbench-source/`：训练/评测 prompt、labels、splits、proxy specs 和数据集元数据，不含视频；
- `plans/`：之前的 Director、轨迹数据集、real proxy、五轮/六轮训练、VBench 和三臂实验计划；
- `representative-results/`：四种 action variant 各一个 case 的三臂真实 MP4、评估 JSON、盲评 request/review/contact sheet；
- `summary/`：三臂汇总统计、配对 delta、95% bootstrap CI、14 维均值、逐 case表和曲线。

## 有意排除

- 完整 `out/training/multi-five-rounds-v1`（约 12 GB）；
- 三臂实验中未选中的其余视频和 Blender 工程文件；
- `.venv`、pytest 临时目录、`__pycache__`、`.pyc`、`.git` 和系统缓存；
- 外部 API key、endpoint 配置和网络响应原文。

代表性视频均来自真实 `D:\blender\blender.exe` 渲染；视觉评分使用当时可用的 Codex 本地帧分析 fallback，并在 review JSON 中标明其 frame-only 限制。
