# 目标 Ubuntu 环境方案（尚未执行）

目标：Ubuntu 24.04.4、RTX PRO 6000 Blackwell 96GB ×1、Ryzen 9950X、128GB RAM。
本文是部署方案，不是经过目标 GPU 验证的环境锁文件。本机 Windows 不创建 Conda 环境。

## 环境划分

| 环境 | 用途 | 初步方案 |
|---|---|---|
| `lingbot-train` | 训练、模型推理服务 | Python 3.12、PyTorch 2.8.0 CUDA 12.8 构建；其余版本优先遵循官方脚本 |
| `robotwin-sim` | SAPIEN、规划器、渲染、闭环客户端 | 固定 RoboTwin 版本要求的 Python/依赖，单独验证 Blackwell/Vulkan/CUDA |
| 转换环境 | 原始 HDF5 → LeRobot | 先选转换器和 LeRobot v2.1/v3.0；有依赖冲突时独立建环境 |

驱动显示 CUDA 13.0 不代表已安装相应 Toolkit。先保留 580 驱动，验证 PyTorch cu128；需要编译扩展时，再检查/准备匹配的 Toolkit。

## 实施和验收顺序

1. 核对 SSD 空间、GPU 空闲状态、已有 Conda/Python、`nvcc` 和 Vulkan，记录实际版本。
2. 参考根目录 `tools/create_train_env.sh` 建训练环境，显式检查 Torch CUDA 构建和 `sm_120` 支持。
3. 对 FlashAttention 2、FlexAttention、fused MoE 做真实前向/反向测试，不能只看 `import` 成功。优先适配依赖或等价算子，保持模型结构及权重语义。
4. 建独立仿真环境，验收离屏渲染、相机 RGB、机器人和规划器；LingBot 训练不需要安装全部 pi0/JAX 训练环境。
5. 按数据文档先转换一条示范、一个任务，通过后再转换 50 个任务。
6. 基础模型启动 FP32 推理并跑少量仿真。基础模型成功率低，不单独说明部署失败。
7. 使用全 50 任务 clean 清单，先跑几步真实更新，再约 100 步，验证保存、加载和断点恢复。
8. 记录依赖精确版本、显存峰值和稳定耗时，再决定第一版训练长度。

环境通过后导出实际依赖记录（如 Conda history 与 pip freeze），提交前检查是否含令牌、私有索引凭据或本机绝对路径。当前不生成假定已验证的锁文件。

## 第一版单卡训练起点

| 设置 | 待测起点 |
|---|---|
| 初始化 | 固定版本 `lingbot-vla-v2-6b` |
| 结构、MoE、深度/视频监督 | 保留官方设置 |
| 优化器 | 原配置 `muon`，不用要求多卡的 `dist_muon` |
| `micro_batch_size` | 1 |
| 梯度累积 | 短测试 1；稳定后尝试 8/16，并同步 global batch |
| 梯度检查点 | 开启 |
| 编译加速 | 初次排错关闭，稳定后比较 |
| 保存 | 短测试验证恢复状态和推理权重导出 |

这不是已验证可在 96GB 上运行的配方。单卡 FSDP 不能分摊状态；若显存不足优先评估 activation offload。参数/优化器 CPU 卸载需核对后端，不能假设某个开关在单卡自动生效。
学习率和总步数根据有效批量和短跑结果确定，不直接套官方 32 卡批量及 50,000 步耗时。

训练、评测分阶段使用同一张卡；推理以 FP32、一个服务、一路仿真起步。最终按组委会当前模板整理评测、代码、权重和复现报告，小规模自测不能写成正式评测已完成。

参考：[官方训练说明](https://github.com/Robbyant/lingbot-vla-v2/blob/main/configs/vla/Training_Config.md)、[PyTorch Blackwell / CUDA 12.8](https://pytorch.org/blog/pytorch-2-7/)。
