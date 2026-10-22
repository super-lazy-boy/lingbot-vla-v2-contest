# 正式训练与评测方案（蚂蚁灵波比赛，2026-10-22 起草）

> **For agentic workers:** 本方案为操作型计划（训练/评测/提交），任务间有硬时间依赖，按 Task 0→7 顺序执行。步骤使用 checkbox 跟踪。

**Goal:** 在 2026-10-26 24:00 截止前，用 50 任务 clean 数据（2500 条示范）在单张 RTX PRO 6000 (96GB) 上完成正式训练 → 闭环评测 → 打包提交，产出可复现的参赛材料。

**Architecture:** 沿用冒烟验证过的链路：`train.sh + robotwin_clean.local.yaml` 单卡训练（FSDP2/Muon/双教师蒸馏）→ 异步导出 hf_ckpt → 官方 `start_robotwin_infer_and_eval.sh` FP32 闭环评测 → 按天池官方模板打包提交。

**Tech Stack:** torch 2.8.0+cu128、LingBot-VLA 2.0 (6B)、RoboTwin 2.0 仿真、LeRobot v2.1 数据集。

**Spec:** 赛制调研 `/mnt/zaq/lingbot-vla-v2-contest-findings.md`；环境台账 `/mnt/zaq/lingbot-workspace/cache/setup_report.md`。

## Global Constraints（规则 + 机器硬约束）

- 训练数据仅 50 任务 clean（`assets/training_data/robotwin_clean.txt`），norm stats 必须 clean-only（`assets/norm_stats/robotwin_clean.json`，已 pin）
- 基座必须 `lingbot-vla-v2-6b`；禁止 randomized 数据训练；仅 BC；最终推理仅一份权重
- 正式训练必须经用户明确批准后方可启动（记忆 contest-training-requires-approval）
- 单卡 96GB，/mnt 机械盘 3.0TB 空闲；评测客户端必须全无代理运行
- 截止 2026-10-26 24:00；初赛全程仅 3 次提交机会

## 时间预算（当前 2026-10-22 15:13，剩余 ~105h）

| 阶段 | 时间窗 | 时长 |
|---|---|---|
| Task 1 预检 + 启动 | 10-22 16:00–17:00 | ~1h |
| Task 2 正式训练（含 4 次暂停评测 ~6h） | 10-22 17:00 – 10-24 18:00 | ~49h |
| Task 6 最终评测（2500 eps） | 10-24 19:00 – 10-25 中午 | ~12–16h |
| Task 7 打包 + 提交 | 10-25 下午 – 10-26 | ~1 天缓冲 |

**实测依据**：micro_batch=1 冒烟 2.2s/it、峰值 55.5GB；评测单 slot 失败 episode ~54s（lift_pot 100 eps 实测 90 分钟）。micro2×accum8 预计 20–30s/步（compile 开/关），49h 窗口净训练 ~43h → 预计 ~5k–7k 步。

## 训练配置决策（推荐值，含回退）

| 参数 | 值 | 依据 | 回退 |
|---|---|---|---|
| 输出目录 | `runs/robotwin_clean_formal`（全新） | 冒烟目录 batch/config 不一致，放弃 step-20 续训 | — |
| micro_batch_size | 2 | batch1 峰值 55.5GB，batch2 ≈ 65GB 可容 | OOM → micro 1 |
| global_batch_size | 16（accum 8，代码自动推导） | 单卡能承受的最大稳定批量 | micro1 → accum 16 |
| lr / lr_min | 1e-4 → 5e-5 cosine | 保持官方值；Muon 对批量不敏感，冒烟 batch1 下 loss 正常下降 | 1000 步 gate 不过 → 2e-5 重启 |
| max_steps | 10000 | 时间裁剪，实际 ~5k–7k 步后硬停 | — |
| save_steps | 2000 | 2k/4k/6k/8k 阶梯供里程碑评测选点 | — |
| use_compile | 预检 3 步冒烟后开启 | 预计 1.3–1.5× 加速 | 失败 → 关（~5k 步） |
| 其余 | 全部沿用冒烟配置 | FP32 动作专家、梯度检查点、fsdp2、muon、深度/DINO-Video 蒸馏不变 | — |

配置文件：`/mnt/zaq/lingbot-workspace/cache/robotwin_clean.formal.yaml`（在 local.yaml 基础上改上述字段，norm_stats 保持 pin）。

---

### Task 0: 用户确认（阻塞，未确认不启动）

- [ ] 批准启动正式训练
- [ ] GPU 占用进程处置：PID 1804465 `rsl_rl/train.py Rail-Flat-Arm-IK-G1-v0`（已跑 3.5h，6.2GB 显存、68% 利用率）——等待 / 停止 / 共享
- [ ] 确认删除冒烟检查点 237GB（global_step_10/11/20，已无价值；磁盘尚余 3TB，非紧急）
- [ ] 评测规模：每任务每 setting 25 episodes（官方允许未测满提交，最终成绩以官方复测为准）——确认
- [ ] 提交次数策略：10-26 提交 1 次，保留 2 次备用——确认

### Task 1: 预检（~1h）

- [ ] 新建 `cache/robotwin_clean.formal.yaml`（micro 2 / global 16 / max_steps 10000 / save_steps 2000 / output formal）
- [ ] compile 3 步冒烟：`env -u ALL_PROXY -u all_proxy bash train.sh tasks/vla/train_lingbotvla.py cache/robotwin_clean.formal.yaml --train.max_steps 3 --train.save_steps 100000`（scratch 输出目录），无 crash 且记录 StepTime
- [ ] 清理 GPU 占用（按 Task 0 决定）；删除冒烟检查点（确认后）
- [ ] 确认磁盘 ≥2TB 空闲、`nvidia-smi` 可用显存 ≥85GB

### Task 2: 启动正式训练 + 监控

- [ ] `nohup` 后台启动（去代理、离线 HF，同 run_train_smoke.sh 模式），日志 `lingbot-workspace/logs/train_formal.log`
- [ ] 启动后 10 分钟内确认：日志打印 micro/global batch 符合预期、无报错
- [ ] 每 1–2h 巡检：loss 下降、无 NaN、显存 <90GB、StepTime 稳定、磁盘 >1TB
- [ ] 1000 步 gate：loss_vla 必须显著低于初始值；否则停训 → LR 改 2e-5 → 从基座重启
- [ ] 保存 2k/4k/6k/8k 点各自 `Success rate` 里程碑评测记录（见 Task 4）

### Task 3: eval client 支持 test_num 覆盖（小改动，随提交包交付）

- [ ] `experiment/robotwin/eval_policy_client_lingbotvla.py:170`：`test_num = int(os.environ.get("EVAL_TEST_NUM", "100"))`（默认 100 不变；208 行专家预检路径不动）
- [ ] 验证：单任务 `EVAL_TEST_NUM=5` 跑通，日志出现 `0/5`、`5/5` 统计行，共 5 个 episode 视频
- [ ] commit：`feat(robotwin): allow EVAL_TEST_NUM to override per-task eval episodes`（代码材料提交时随包交付，写入复现报告）

### Task 4: 里程碑评测（2k/4k/6k/8k 四点）

- [ ] 暂停协议：等当前 step 的 DCP + hf_ckpt 导出完成（日志 `[async_hf] HF checkpoint finished`）→ 停训练 → 评测 → 原命令重启（自动 resume 最新 DCP）
- [ ] 每点：`start_robotwin_infer_and_eval.sh --model_path <该点 hf_ckpt> --eval_workdir third_party/RoboTwin --task_config demo_clean --num_tasks 2 --num_gpus 1 --num_per_gpu 2`（前 2 任务 lift_pot、hanging_mug × 100 eps，FP32，无代理环境）
- [ ] 记录四点成功率；若 8k 点成功率持平/下降 → 提前硬停进入 Task 5
- [ ] 每次暂停后确认训练 resume 成功且 loss 续降

### Task 5: 硬停 + 选点（10-24 18:00）

- [ ] 停止训练（安全点：最近一次 save 完成后），保留全部 hf_ckpt
- [ ] 按里程碑成功率选最终 checkpoint（默认最后一点）

### Task 6: 最终评测

- [ ] demo_clean：50 任务 × `EVAL_TEST_NUM=25`，`--num_gpus 1 --num_per_gpu 2` 起步；`nvidia-smi` 确认有富余则升 3 slots
- [ ] demo_randomized：同上
- [ ] 时间富余则提高到 50/task；产出 stats.txt，按官方模板填 results.json（实际测试次数与成功次数）
- [ ] 评测全程无代理；每 2h 巡检成功率与剩余时间

### Task 7: 提交包（10-25 下午 – 10-26）

- [ ] 官方结构：`01_评测结果/results.json`、`02_代码材料/`（README.md、train.sh、eval.sh、configs/、src/）、`03_模型权重/checkpoint/`（最终 hf_ckpt）、`04_复现与调优说明/reproduction_report.md`
- [ ] 复现报告要素：训练命令与配置差异（对官方 robotwin.yaml 的改动逐条列出）、norm stats 生成方式、eval.sh 参数与评测环境（FP32、无代理）、硬件、成功率
- [ ] 自检：解压 zip → 权重可加载 → train.sh/eval.sh 参数可解析
- [ ] 10-26 提交 1 次（保留 2 次备用）

## Review Focus（最可能翻车的点）

1. **评测时长失控**（54s/episode 实测）→ Task 3 的 EVAL_TEST_NUM 是唯一解药，必须最先落地
2. **GPU 争抢**（rsl_rl 68% 利用率会拖慢训练与评测）→ Task 0 必须明确处置
3. **compile 在 FSDP2+flex_cached 组合上未验证** → Task 1 预检 gate，失败即关
4. **暂停-恢复破坏训练状态** → 只在 save 完成后暂停；resume 后确认 loss 续降
5. **提交包缺复现细节**（官方明确「无法复现后果自负」）→ Task 7 逐条对照 findings §4.3 清单
