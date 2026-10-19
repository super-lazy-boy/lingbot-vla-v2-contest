# 2026 比赛工作区

本目录管理比赛准备、数据来源和后续实验记录，保留官方代码历史和许可证。
当前阶段：**本机只准备代码和下载清单，数据与模型到目标 Ubuntu 电脑下载；本机不安装训练或仿真环境。**

## 文件入口

| 内容 | 位置 |
|---|---|
| 50 个 clean 任务、固定版本、大小、SHA-256 | `data_manifest.json` |
| 独立校验表、任务名单 | `SHA256SUMS`、`tasks.txt` |
| 代码、模型、数据版本 | `sources.lock.json` |
| 数据下载、断点续传、校验 | `tools/prepare_data.py` |
| 模型下载计划及目标机下载入口 | `tools/download_models.py` |
| 固定版本 RoboTwin 源码 | `../third_party/RoboTwin`，Git 子模块 |
| Ubuntu / Conda / GPU 验收方案 | [ENVIRONMENT_PLAN.md](ENVIRONMENT_PLAN.md) |
| 转换与训练前检查 | [DATA_PREPARATION.md](DATA_PREPARATION.md) |
| 分支、实验、上游同步约定 | [GIT_WORKFLOW.md](GIT_WORKFLOW.md) |

代码和清单已准备，不等于已部署、已转换数据或已验证 GPU 训练。

## 比赛范围

- 从 `robbyant/lingbot-vla-v2-6b` 基础模型初始化，不从 `-robotwin` 后训练权重初始化。
- 训练使用 Aloha-AgileX 的 50 个任务，每任务 50 条 clean 示范，共 2500 条。
- randomized 不加入训练或归一化统计。第一版保留模型结构和原有深度/视频监督。
- Git 不保存数据 ZIP、模型权重、checkpoint、视频、令牌或本机路径配置。
- 规则依据：[官方赛题与数据](https://tianchi.aliyun.com/competition/entrance/532514/information)。如有更新以组委会当前要求为准。

## 目标电脑目录

代码放 NVMe SSD，例如 `$HOME/projects/lingbot-vla-v2-contest`；另设 SSD 工作目录 `$HOME/lingbot-workspace`：

```text
lingbot-workspace/
  datasets/robotwin2/raw/<task>/demo_clean.zip
  datasets/robotwin2/extracted/
  datasets/robotwin2/lerobot/
  models/lingbot-vla-v2-6b/
  models/Qwen3-VL-4B-Instruct/
  models/moge-2-vitb-normal/
  runs/
  cache/
```

原始 ZIP、旧 checkpoint 和视频可备份到 `/mnt` 的机械硬盘；正在训练的数据和当前输出优先放 SSD。
预留 500GB～1TB SSD 是工作空间规划，不是已测出的实际占用。

## 目标电脑下载数据

先在目标电脑的 SSD 克隆仓库，再初始化指定子模块：

```bash
mkdir -p "$HOME/projects"
git clone https://github.com/super-lazy-boy/lingbot-vla-v2-contest.git "$HOME/projects/lingbot-vla-v2-contest"
cd "$HOME/projects/lingbot-vla-v2-contest"
git submodule update --init third_party/RoboTwin
```

这是私有仓库，目标电脑需使用有权访问该仓库的 GitHub 登录；不要把令牌写入 URL 或提交到代码中。

以下命令只要求 Python 3.10+，不要求先安装 Torch、Conda 或 Hugging Face 客户端：

```bash
python3 competition/tools/prepare_data.py status --root "$HOME/lingbot-workspace/datasets/robotwin2"
python3 competition/tools/prepare_data.py download --root "$HOME/lingbot-workspace/datasets/robotwin2"
python3 competition/tools/prepare_data.py verify --root "$HOME/lingbot-workspace/datasets/robotwin2"
```

只下载一个任务用于验收：

```bash
python3 competition/tools/prepare_data.py download --root "$HOME/lingbot-workspace/datasets/robotwin2" --task adjust_bottle
```

中断后重跑同一命令即可续传；已有完整文件仍会校验 SHA-256。完整但损坏的临时文件保留为 `.corrupt-*`，可在确认不需要后人工清理。
50 个 ZIP 合计 **23,375,659,439 字节，约 23.38GB / 21.77GiB**；工具额外预留 5GiB 下载余量，解压和转换另算。工具只下载原始 ZIP，不会解压或生成 LeRobot 数据。

## 目标电脑准备模型

先查看计划，不下载：

```bash
python3 competition/tools/download_models.py --root "$HOME/lingbot-workspace/models"
```

目标环境安装好 `huggingface_hub` 后，再显式下载：

```bash
python competition/tools/download_models.py --root "$HOME/lingbot-workspace/models" --download
```

基础模型仓库内已包含 `depth/` 和 `dino_video/`。Qwen 默认只下载配置、分词器和处理器，VLA checkpoint 已包含 VLM 权重；确需完整独立 Qwen 模型时加 `--qwen-full`。

## 本机验证

以下验证不联网、不安装依赖、不启动训练：

```text
python -m unittest discover -s competition/tests -v
python competition/tools/prepare_data.py status
python competition/tools/download_models.py --root local_models
```

## 上游来源

- [LingBot-VLA](https://github.com/Robbyant/lingbot-vla-v2)：Apache-2.0，保留根目录 `LICENSE` 与 Git 历史。
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)：子模块保留自己的许可证；按 [LingBot 官方说明](../experiment/robotwin/README.md) 固定到 `13c3c47ff4312dd62484bcd51be034af55c062d1`。
- [数据集](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)：具体发布版本和校验值见清单。
