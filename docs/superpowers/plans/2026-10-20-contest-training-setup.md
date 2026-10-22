# 比赛训练环境准备实施计划（目标机部署）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本机（Ubuntu 24.04.4 / RTX PRO 6000 Blackwell 96GB / 128GB RAM）完成蚂蚁灵波挑战赛的训练准备：代码就绪、50 任务 clean 数据集下载校验并转换为 LeRobot、模型权重下载、训练与仿真环境搭建，并通过端到端冒烟验证。

**Architecture:** 严格按仓库内既有规格文档执行——代码由 `/mnt/zaq/lingbot-vla-v2-contest` 仓库管理（codex/* 分支 + upstream 同步），数据与模型全部落在 `/mnt/zaq/lingbot-workspace/`（用户要求：下载都在 /mnt/zaq 内），环境按 ENVIRONMENT_PLAN.md 划分为 `lingbot-train`（PyTorch 2.8.0 cu128）与 `robotwin-sim`（SAPIEN 仿真），转换沿用官方 RoboTwin 固定 commit 的脚本链路。

**Tech Stack:** conda/miniforge 26.1.1、Python 3.12、PyTorch 2.8.0 (cu128)、flash-attn 2.8.3、LeRobot v0.4.2、RoboTwin 子模块 @13c3c47、SAPIEN、huggingface_hub。

**Spec:** `competition/README.md`、`competition/ENVIRONMENT_PLAN.md`、`competition/DATA_PREPARATION.md`、`competition/GIT_WORKFLOW.md`、`competition/sources.lock.json`、`competition/data_manifest.json`。

## Global Constraints（取自 spec，逐条）

- 基座权重只能是 `robbyant/lingbot-vla-v2-6b` @ `11c703bf6a5c1f45b3b69168482da11fdbba53d7`，禁止 `-robotwin` 后训练权重。
- 训练数据只能是 `TianxingChen/RoboTwin2.0` @ `3dc3b798668feb99ac61cc9086d84cbcc3d79186` 的 `dataset/<task>/demo_clean.zip`，50 任务 × 50 条 = 2500 条；randomized 不进训练也不进归一化统计。
- RoboTwin 固定 commit `13c3c47ff4312dd62484bcd51be034af55c062d1`（子模块）。
- `lingbot-train` 环境：Python 3.12、torch==2.8.0；保留 580 驱动（CUDA 13.0），验证 cu128 与 sm_120。
- Git 边界：数据 ZIP、模型权重、checkpoint、视频、令牌、本机绝对路径一律不进提交；`.gitattributes` 固定 LF。
- 工作区根目录 `/mnt/zaq/lingbot-workspace/`；代码仓库 `/mnt/zaq/lingbot-vla-v2-contest`。
- 任何代码改动走 `codex/<主题>` 分支（GIT_WORKFLOW.md），验证后合并 main。

## Review Focus（最可能翻车的点，每项都有对应任务的验证）

1. **flash-attn 2.8.3 在无 nvcc 的机器上从源码编译失败**（本机未装 CUDA Toolkit）→ Task 3 先探测 wheel、失败则 conda 装 `cuda-nvcc` 后重试。
2. **/mnt 是机械盘**（docs 规划 SSD，用户要求 /mnt/zaq）→ Task 0 记录磁盘类型，Task 6 记录训练步耗时；若过慢再向用户提议挪热数据到 NVMe + 软链。
3. **新版数据集 JPEG 与旧转换脚本 RGB 语义错位（红蓝通道交换）**→ Task 5 单任务样本解码比对，训练/推理保持同一 RGB 语义。
4. **SAPIEN 在 Blackwell 上离屏渲染失败（Vulkan/CUDA）**→ Task 4 单独验收，渲染并保存相机 RGB。
5. **单卡 FSDP2 + FP32 动作专家在 96GB 上显存不足**→ Task 6 短训实测峰值显存（micro_batch 1 + 梯度检查点 + 初次关 compile）。
6. **randomized 数据混入训练清单或统计**→ Task 5 统计只算 clean；manifest 工具本身拒绝非 clean 条目。
7. **HF 下载慢/中断**→ Task 1 单任务测速；工具带断点续传与 SHA-256 校验；必要时与用户确认切换 hf-mirror。
8. **提交污染**：环境导出/实验记录含令牌或绝对路径 → Task 7 检查后才落仓库，敏感记录只留在工作区。

---

### Task 0: 仓库与工作区基线

**Files:**
- Modify: 无（仅 `git restore` 一个行尾噪声文件）
- Create: `/mnt/zaq/lingbot-workspace/{datasets/robotwin2/{raw,extracted,lerobot},models,runs,cache,logs}`

**Interfaces:**
- Produces: 工作区目录树（Task 1–7 全部使用）；`upstream` remote；干净的 working tree。

- [ ] **Step 1: 确认 HEAD 与工作区状态**

```bash
cd /mnt/zaq/lingbot-vla-v2-contest
git log --oneline -1          # 期望: ed2b147 Prepare competition workspace and pinned clean data downloads
git status -sb               # 期望: 仅 M lingbotvla/models/vla/vision_models/MoGe/SUPPORT.md
```

- [ ] **Step 2: 确认 SUPPORT.md 改动只是行尾（LF 归一化），并还原**

```bash
git diff --ignore-space-at-eol --stat   # 期望: 无输出（纯行尾差异）
git restore lingbotvla/models/vla/vision_models/MoGe/SUPPORT.md
git status -sb                            # 期望: working tree clean
```

- [ ] **Step 3: 添加 upstream 并核验子模块**

```bash
git remote add upstream https://github.com/Robbyant/lingbot-vla-v2.git
git remote -v                                        # 期望: origin=比赛仓库, upstream=Robbyant/lingbot-vla-v2
git submodule status                                 # 期望: third_party/RoboTwin 前缀为空格，commit 13c3c47（已初始化）
```

- [ ] **Step 4: 创建工作区目录树并记录磁盘信息**

```bash
mkdir -p /mnt/zaq/lingbot-workspace/datasets/robotwin2/{raw,extracted,lerobot} \
         /mnt/zaq/lingbot-workspace/{models,runs,cache,logs}
df -h /mnt/zaq | tee /mnt/zaq/lingbot-workspace/cache/disk_before.txt
lsblk -d -o NAME,ROTA,SIZE /dev/sda            # ROTA=1 为机械盘，记录到 setup_report
```

- [ ] **Step 5: 跑仓库自带的准备工具测试（系统 Python 3.12，纯标准库，不联网）**

```bash
cd /mnt/zaq/lingbot-vla-v2-contest
python3 -m unittest discover -s competition/tests -v
python3 competition/tools/prepare_data.py status   # 期望: 打印 50 个 clean 档案与 Disk space: OK
python3 competition/tools/download_models.py --root /mnt/zaq/lingbot-workspace/models
```

期望：测试全过；status 显示 `Clean archives: 50; compressed size: 23.38 GB`、`Disk space: OK`；download_models 只打印计划不下载。

---

### Task 1: 下载 50 任务 clean 数据集（23.38GB，可后台）

**Files:**
- Create: `/mnt/zaq/lingbot-workspace/datasets/robotwin2/raw/<task>/demo_clean.zip` × 50

**Interfaces:**
- Consumes: Task 0 的 `raw/` 目录与 `competition/tools/prepare_data.py`（stdlib，无需任何环境）。
- Produces: 50 个 SHA-256 已校验的原始 ZIP；测速数据供后续决定是否切镜像。

- [ ] **Step 1: 单任务测速（adjust_bottle，292MB）**

```bash
cd /mnt/zaq/lingbot-vla-v2-contest
python3 competition/tools/prepare_data.py download \
  --root /mnt/zaq/lingbot-workspace/datasets/robotwin2 --task adjust_bottle
```

期望输出以 `DOWNLOADED adjust_bottle (SHA-256 verified)` 结尾。用 `time` 包裹，记录速率到 cache/setup_report。

- [ ] **Step 2: 决定下载源**

直连 HF 速率 ≥ 2MB/s/连接 → 保持默认。若明显更慢：向用户确认后切换 hf-mirror（工具 URL 硬编码 huggingface.co，需在 `codex/env-setup` 分支给 `prepare_data.py` 加 `--base-url` 参数并同步改 URL 拼接处；改完先跑 `python3 -m unittest discover -s competition/tests -v`）。用户未确认前不切。

- [ ] **Step 3: 全量下载（后台）**

```bash
cd /mnt/zaq/lingbot-vla-v2-contest
nohup python3 competition/tools/prepare_data.py download \
  --root /mnt/zaq/lingbot-workspace/datasets/robotwin2 --workers 6 \
  > /mnt/zaq/lingbot-workspace/logs/data_download.log 2>&1 &
```

用 Monitor 观察 `logs/data_download.log`，过滤 `DOWNLOADED|VERIFIED|ERROR`（覆盖成功与失败两种终态；`tail -f ... | grep -E --line-buffered`）。工具支持断点续传：中断后重跑同一命令。

- [ ] **Step 4: 全量校验**

```bash
python3 competition/tools/prepare_data.py verify \
  --root /mnt/zaq/lingbot-workspace/datasets/robotwin2
```

期望: `All 50 selected archives passed SHA-256 verification.`
核对 `ls raw | wc -l` == 50，`du -sh raw` ≈ 21.8GiB，与 `competition/tasks.txt` 名单一致。

---

### Task 2: 下载模型权重（约 15–20GB）

**Files:**
- Create: `/mnt/zaq/lingbot-workspace/models/{lingbot-vla-v2-6b,Qwen3-VL-4B-Instruct,moge-2-vitb-normal}/`

**Interfaces:**
- Consumes: `competition/tools/download_models.py` + `huggingface_hub`（base conda 已有 hf 1.21.0）。
- Produces: Task 5/6 使用的基座权重、Qwen processor、MoGe 教师权重。

- [ ] **Step 1: 确认 huggingface_hub 可用**

```bash
python3 -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

期望打印版本号（base 环境）。若 ImportError：`python3 -m venv /mnt/zaq/lingbot-workspace/tools-venv && /mnt/zaq/lingbot-workspace/tools-venv/bin/pip install huggingface_hub` 后改用该 python。

- [ ] **Step 2: 先看计划再下载**

```bash
cd /mnt/zaq/lingbot-vla-v2-contest
python3 competition/tools/download_models.py --root /mnt/zaq/lingbot-workspace/models
```

期望: 打印 base/qwen/moge 三个 repo@revision 与目标目录，Qwen 标注只取配置/分词器/处理器；无任何下载。

- [ ] **Step 3: 后台下载（Qwen 默认只取配置文件）**

```bash
nohup python3 competition/tools/download_models.py \
  --root /mnt/zaq/lingbot-workspace/models --download \
  > /mnt/zaq/lingbot-workspace/logs/models_download.log 2>&1 &
```

完成标志：进程退出且 `du -sh /mnt/zaq/lingbot-workspace/models/*` 均有体量。hf-mirror 兜底同 Task 1 Step 2 的确认流程（`snapshot_download` 尊重 `HF_ENDPOINT` 环境变量）。

- [ ] **Step 4: 关键文件核验**

```bash
ls /mnt/zaq/lingbot-workspace/models/lingbot-vla-v2-6b/        # 期望含 model 权重、config、depth/、dino_video/
ls /mnt/zaq/lingbot-workspace/models/Qwen3-VL-4B-Instruct/     # 期望含 config.json、tokenizer*、preprocessor_config.json
ls /mnt/zaq/lingbot-workspace/models/moge-2-vitb-normal/       # 期望含模型 checkpoint 与配置
```

记录各目录大小到 cache/setup_report。快照按 revision 固定，revision 一致性由 `snapshot_download(revision=...)` 保证，记录 revision 字符串备查。

---

### Task 3: lingbot-train 训练环境 + GPU 算子冒烟

**Files:**
- Create: `competition/tools/gpu_smoke.py`（新文件，内容见 Step 5）
- Modify: 无

**Interfaces:**
- Consumes: Task 2 的权重路径（冒烟测试不依赖权重，仅依赖已装环境）。
- Produces: conda env `lingbot-train`；`gpu_smoke.py` 全 PASS；环境依赖记录 `cache/lingbot-train-env.txt`。

- [ ] **Step 1: 探测 flash-attn 是否已有 cp312/cu128 轮子（决定是否先装 nvcc）**

```bash
conda run -n base pip index versions flash-attn 2>/dev/null | head -5
# 或安装时看 pip 是否直接下载 wheel 而非源码包；本机 nvcc: command not found（已确认）
```

- [ ] **Step 2: 用官方脚本建环境（脚本自带 torch 版本断言）**

```bash
cd /mnt/zaq/lingbot-vla-v2-contest
bash tools/create_train_env.sh --env-name lingbot-train
```

期望脚本以 `Environment ready: lingbot-train` 结束，期间多次打印 `torch 2.8.0 cuda 12.8 available True`。
若在 flash-attn 阶段因缺 nvcc 编译失败：`conda install -n lingbot-train -c conda-forge cuda-nvcc=12.8 -y`，然后 `bash tools/create_train_env.sh --env-name lingbot-train --resume`。仍失败则尝试官方预编译 wheel（flash-attn GitHub Releases 的 cu128 轮子），并在 setup_report 记录实际方案。

- [ ] **Step 3: 验证 sm_120 与 CUDA 可用**

```bash
conda run -n lingbot-train python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_capability(0))   # 期望 (12, 0)
PY
```

- [ ] **Step 4: 写 gpu_smoke.py 骨架文件**

创建 `competition/tools/gpu_smoke.py`（内容如下；若 Qwen2TokenMoeBlock 构造字段与默认 config 不符，按 `lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py:222-363` 的实际字段名修正后再跑）：

```python
#!/usr/bin/env python3
"""Blackwell 冒烟：FlashAttention2 / FlexAttention / fused MoE 的真实前向+反向。"""
import torch

def check(name, fn):
    try:
        fn()
        print(f"PASS {name}")
    except Exception as e:
        print(f"FAIL {name}: {e!r}")
        raise

def _fwd_bwd(out, label):
    assert torch.isfinite(out).all(), f"{label} output not finite"
    out.sum().backward()

def flash_attn_fwd_bwd():
    from flash_attn import flash_attn_func
    torch.manual_seed(0)
    q = torch.randn(1, 2048, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=True)
    _fwd_bwd(out, "flash_attn")
    assert q.grad is not None and torch.isfinite(q.grad).all()

def flex_attention_fwd_bwd():
    from torch.nn.attention.flex_attention import flex_attention
    torch.manual_seed(0)
    q = torch.randn(1, 8, 2048, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    def causal(score, b, h, q_idx, kv_idx):
        return torch.where(q_idx >= kv_idx, score, -float("inf"))
    out = flex_attention(q, k, v, score_mod=causal)
    _fwd_bwd(out, "flex_attention")
    assert q.grad is not None and torch.isfinite(q.grad).all()

def fused_moe_fwd_bwd():
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2TokenMoeBlock
    torch.manual_seed(0)
    cfg = LingbotVLAV2Config()
    block = Qwen2TokenMoeBlock(cfg).to("cuda")
    x = torch.randn(2, 128, cfg.expert_hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = block(x)
    _fwd_bwd(out, "fused_moe")
    assert x.grad is not None and torch.isfinite(x.grad).all()

if __name__ == "__main__":
    assert torch.cuda.is_available()
    check("flash_attn fwd+bwd", flash_attn_fwd_bwd)
    check("flex_attention fwd+bwd", flex_attention_fwd_bwd)
    check("fused_moe fwd+bwd", fused_moe_fwd_bwd)
    print("gpu_smoke: ALL PASS")
```

- [ ] **Step 5: 运行冒烟测试**

```bash
conda run -n lingbot-train python competition/tools/gpu_smoke.py
```

期望: 三行 PASS + `gpu_smoke: ALL PASS`。flex_attention 首次运行会触发编译，等待数秒属正常。

- [ ] **Step 6: 导出依赖记录并提交冒烟脚本**

```bash
conda list -n lingbot-train --explicit > /mnt/zaq/lingbot-workspace/cache/lingbot-train-env.txt
conda run -n lingbot-train pip freeze > /mnt/zaq/lingbot-workspace/cache/lingbot-train-pip.txt
grep -iE "token|password|secret" /mnt/zaq/lingbot-workspace/cache/lingbot-train-*.txt && echo "CHECK FAILED" || echo "no secrets"
cd /mnt/zaq/lingbot-vla-v2-contest
git switch -c codex/env-setup
git add competition/tools/gpu_smoke.py
git commit -m "feat(competition): add Blackwell GPU smoke test (FA2/FlexAttention/MoE)"
```

期望: `no secrets`；提交落在 `codex/env-setup` 分支，main 不动。

---

### Task 4: robotwin-sim 仿真环境 + 离屏渲染验收

**Files:**
- Create: `/mnt/zaq/lingbot-workspace/logs/sim_smoke.log`、`/mnt/zaq/lingbot-workspace/cache/sim_camera_check.png`

**Interfaces:**
- Consumes: Task 0 子模块 `third_party/RoboTwin`（已固定 commit）。
- Produces: conda env `robotwin-sim`；离屏渲染出的相机 RGB 图；仿真可用结论。

- [ ] **Step 1: 读安装说明（官方文档 + 仓库自带的 _install.sh）**

```bash
sed -n '1,80p' /mnt/zaq/lingbot-vla-v2-contest/third_party/RoboTwin/script/_install.sh
```

并用 WebFetch 读 `https://robotwin-platform.github.io/doc/usage/robotwin-install.html`，确定官方要求的 Python 版本与 SAPIEN 版本。

- [ ] **Step 2: 建环境并按官方步骤安装**

按 Step 1 得到的官方流程创建 `robotwin-sim`（python 版本以官方为准，不要照搬 lingbot-train 的 3.12；若官方要求 conda env 文件则用之）。`pip` 安装时如遇网络慢，用与 Task 1 相同的镜像确认流程。

- [ ] **Step 3: 离屏渲染冒烟**

参考 `third_party/RoboTwin/script/eval_policy_client.py` 的环境实例化方式，写一次性脚本（放 `/mnt/zaq/lingbot-workspace/cache/sim_smoke.py`，不进仓库）：启动 SAPIEN GPU 渲染器 → 加载任一任务场景（如 adjust_bottle，task_config 在 `third_party/RoboTwin/task_config/`）→ 步进几帧 → 保存相机 RGB 为 `cache/sim_camera_check.png`。断言：引擎创建成功、`nvidia-smi` 显示该进程占用显存、保存的 PNG 非全黑/全白。

- [ ] **Step 4: 记录并验收**

期望: `sim_camera_check.png` 正常成像；`ls /usr/share/vulkan/icd.d/nvidia_icd.json` 存在（已确认）。把 Python/SAPIEN 版本写进 cache/setup_report。若渲染失败：按 diagnosing-bugs 思路查 Vulkan ICD/EGL，不降级 CPU 渲染蒙混过关——CPU 渲染会拖慢评测且与官方复测环境不符。

---

### Task 5: 数据转换（HDF5 → LeRobot v2.1，先 1 后 50）

**Files:**
- Create: `competition/tools/extract_data.py`（安全解压，内容见 Step 2）
- Create: `/mnt/zaq/lingbot-workspace/datasets/robotwin2/lerobot/<task>/` × 50、`assets/norm_stats/robotwin_clean.json`、`assets/training_data/robotwin_clean.txt`

**Interfaces:**
- Consumes: Task 1 的 50 个 ZIP；Task 4 或 lingbot-train 环境（转换依赖 h5py/cv2/lerobot）。
- Produces: Task 6 训练/评测用的 LeRobot 数据集与 clean-only 归一化统计。

- [ ] **Step 1: 确定转换运行环境**

先试 `lingbot-train`（已含 cv2/lerobot，缺 h5py → `conda run -n lingbot-train pip install h5py`）。若转换脚本 import 冲突，按 ENVIRONMENT_PLAN 单独建 `robotwin-convert` 环境。`XDG_CACHE_HOME` 指向 `/mnt/zaq/lingbot-workspace/cache`（官方 README 要求缓存盘空间充足）。

- [ ] **Step 2: 写安全解压工具**

创建 `competition/tools/extract_data.py`：读取 ZIP 后逐条目检查——拒绝绝对路径、`..`、符号链接（`(zi.external_attr >> 16) & 0o170000 == 0o120000`）、路径穿越；逐任务解压到 `datasets/robotwin2/extracted/<task>/`。检查通过才写文件。

- [ ] **Step 3: 解压并检查单任务 adjust_bottle**

```bash
python3 competition/tools/extract_data.py \
  --root /mnt/zaq/lingbot-workspace/datasets/robotwin2 --task adjust_bottle
```

期望: `extracted/adjust_bottle/` 下与官方布局一致（ZIP 内为 `dataset/adjust_bottle/demo_clean/...`），解压后按官方 README 整理为 `data/adjust_bottle/aloha-agilex_clean_50/`；用 h5py 读一条轨迹：恰好 50 条示范；状态/动作为 14 维（左臂6+左夹爪1+右臂6+右夹爪1）；三路相机（主相机+左右腕）。打开一张图像保存为 PNG 人工/程序比对 RGB 语义（重点核对新版 JPEG 无红蓝通道交换，见 DATA_PREPARATION.md 第 4 条）。

- [ ] **Step 4: 单任务转换**

```bash
cd /mnt/zaq/lingbot-vla-v2-contest/third_party/RoboTwin/policy/pi0
bash process_data_pi0.sh adjust_bottle aloha-agilex_clean_50 50
bash generate.sh ./training_data/demo_clean/ robotwin_clean_adjust_bottle
```

期望: 产出 `processed_data/adjust_bottle-aloha-agilex_clean_50-50/`；LeRobot v2.1 数据落在 `$XDG_CACHE_HOME/huggingface/lerobot/robotwin_clean_adjust_bottle`，随后移动到 `/mnt/zaq/lingbot-workspace/datasets/robotwin2/lerobot/adjust_bottle`。检查该数据集：50 个 episode、帧数对齐、actions/states 形状 `[50, 14]`、语言指令非空、未来帧对齐无越界。

- [ ] **Step 5: 扩展到全部 50 任务**

对 `competition/tasks.txt` 的每个任务循环执行 Step 3–4（含逐任务 episode 数核验）。期望最终 50 个 LeRobot 目录、合计 2500 条示范。生成清单文件：

```bash
for t in $(cat /mnt/zaq/lingbot-vla-v2-contest/competition/tasks.txt); do
  echo "robotwin_clean /mnt/zaq/lingbot-workspace/datasets/robotwin2/lerobot/$t"
done > /mnt/zaq/lingbot-vla-v2-contest/assets/training_data/robotwin_clean.txt
```

- [ ] **Step 6: 计算 clean-only 归一化统计**

按 `lingbotvla/data/vla_data/README.md` 与 `scripts/compute_norm_stats.py` 的用法，仅以 `robotwin_clean.txt` 的 50 个 clean 目录计算统计，输出到 `assets/norm_stats/robotwin_clean.json`。核验 JSON 里没有 randomized 目录的任何痕迹（数据路径与统计条目全部指向 clean 目录）。

- [ ] **Step 7: 提交转换工具**

```bash
cd /mnt/zaq/lingbot-vla-v2-contest
git add competition/tools/extract_data.py assets/training_data/robotwin_clean.txt assets/norm_stats/robotwin_clean.json
git commit -m "feat(competition): safe extract tool and clean-only robotwin data list + norm stats"
```

转换过程日志与版本摘要写入 `/mnt/zaq/lingbot-workspace/cache/conversion_report.md`（含转换器 commit、每任务示范数、统计文件摘要；**不进 Git**）。

---

### Task 6: 端到端冒烟（基座 FP32 推理 + 短训练）

**Files:**
- Create: `configs/vla/robotwin_clean/robotwin_clean.yaml`（基于官方 `configs/vla/robotwin/robotwin.yaml` 改）
- Create: `/mnt/zaq/lingbot-workspace/runs/smoke_*`

**Interfaces:**
- Consumes: Task 2 权重、Task 3 环境、Task 4 仿真、Task 5 数据与统计。
- Produces: 基座推理成功率行（低分正常）；短训练 loss 曲线、checkpoint 保存/恢复、hf_ckpt 导出验证；峰值显存记录。

- [ ] **Step 1: 建 clean 训练配置**

复制官方 `configs/vla/robotwin/robotwin.yaml` 为 `configs/vla/robotwin_clean/robotwin_clean.yaml`，修改：数据列表 → `assets/training_data/robotwin_clean.txt`；norm stats → `assets/norm_stats/robotwin_clean.json`；`micro_batch_size: 1`、梯度累积 1、`gradient_checkpointing` 开启、`torch.compile` 关闭（按 ENVIRONMENT_PLAN.md「第一版单卡训练起点」表）。优化器保持 `muon`。

- [ ] **Step 2: 基座 FP32 推理 + 单任务少量 episode**

按 `experiment/robotwin/README.md` Prerequisites（`SIM_ENV=robotwin-sim`、`INFERENCE_ENV=lingbot-train`、`QWEN3VL_PATH=/mnt/zaq/lingbot-workspace/models/Qwen3-VL-4B-Instruct`），用 `start_robotwin_infer_and_eval.sh` 对 1 个任务跑少量 episode（FP32、1 GPU）：

```bash
cd /mnt/zaq/lingbot-vla-v2-contest/experiment/robotwin
# 参数按脚本 --help 对齐，MODEL_PATH=/mnt/zaq/lingbot-workspace/models/lingbot-vla-v2-6b
```

期望: 推理服务启动、仿真跑通、日志出现 `Success rate: x/100`（基座成功率低属预期，ENVIRONMENT_PLAN 第 6 条）；全程无 CUDA OOM。记录显存峰值。

- [ ] **Step 3: 短训练（约 10 步）**

```bash
cd /mnt/zaq/lingbot-vla-v2-contest
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin_clean/robotwin_clean.yaml \
  --data.norm_stats_file assets/norm_stats/robotwin_clean.json
```

期望: loss 有值且无 NaN/Inf；DCP checkpoint 正常保存；`nvidia-smi` 记录峰值显存；单步耗时记录到 setup_report。

- [ ] **Step 4: 断点恢复 + 推理权重导出**

从 Step 3 的 checkpoint 再续 10 步（验证保存/加载）；确认 hf_ckpt 导出成功，且能被 `deploy/lingbot_vla_v2_policy.py` 加载（起服务后请求一次推理，不要求成功率）。

- [ ] **Step 5: 提交配置**

```bash
git add configs/vla/robotwin_clean/
git commit -m "feat(competition): single-GPU clean training config for robotwin (micro_batch=1)"
```

---

### Task 7: 记录、标签与收尾

**Files:**
- Create: `/mnt/zaq/lingbot-workspace/cache/setup_report.md`（不进 Git）
- Modify: `competition/README.md`（「当前阶段」一句更新，可选）

**Interfaces:**
- Consumes: Task 0–6 的日志与结论。
- Produces: 完整环境记录；`prep-2026-10-20` 标签（仅当 Task 1–6 全部验证通过）；合并回 main 的建议。

- [ ] **Step 1: 汇总 setup_report.md**

汇总：硬件与磁盘类型（Task 0）、下载源与速率（Task 1/2）、环境依赖导出与 flash-attn 安装方式（Task 3）、仿真版本（Task 4）、转换器与统计摘要（Task 5）、峰值显存与步耗时（Task 6）。**检查不含令牌/密钥**（Task 3 Step 6 同款 grep）。

- [ ] **Step 2: 打标签并整理分支**

```bash
cd /mnt/zaq/lingbot-vla-v2-contest
git tag prep-2026-10-20      # 仅在 Task 1–6 全部验证通过后
```

`codex/env-setup` 分支已包含 Task 3/5/6 的提交；向用户确认后合并 main（GIT_WORKFLOW：验证后合并）。合并后 main 只含工具、配置与文档，不含任何数据/权重/日志。

- [ ] **Step 3: 可选文档更新**

将 `competition/README.md`「当前阶段」行改为目标机已部署的现状描述（环境名、工作区路径、验证结论），单独一次提交。

---

## Self-Review 记录

- **Spec 覆盖**：GIT_WORKFLOW（Task 0/3/5/6/7 的分支、upstream、标签、文件边界）✓；ENVIRONMENT_PLAN 验收顺序 1–8（Task 0 磁盘/GPU 记录、Task 3 环境+算子、Task 4 仿真、Task 5 转换、Task 6 基座推理+短训练、Task 7 版本记录）✓；DATA_PREPARATION 转换顺序 1–10（Task 5 各步）✓；competition/README 下载与验证命令（Task 1/2）✓。
- **Placeholder 扫描**：无 TBD/TODO；Task 3 Step 4 与 Task 4 Step 3 的“按实际字段/API 修正”是面对第三方代码接口的明确指引，非空洞描述。
- **类型一致性**：工作区路径全篇统一 `/mnt/zaq/lingbot-workspace/...`；环境名 `lingbot-train`/`robotwin-sim` 与 ENVIRONMENT_PLAN 一致；revision 字符串与 sources.lock.json 逐字一致。
- **Review Focus**：8 条风险均映射到对应 Task（标注在各项内）。
