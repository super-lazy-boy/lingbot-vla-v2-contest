# 数据转换与验收（目标机执行）

当前仅固定 revision、50 个 clean 任务及每个 ZIP 的官方 SHA-256；未下载大文件、未解压、未转换 LeRobot、未计算 clean 统计。

`data_manifest.json` 仅包含 `dataset/<task>/demo_clean.zip`。更新数据版本时同步更新清单、校验表和实验记录，不在同一实验中悄悄混用 `main` 的新文件。

## 版本兼容性

准备时数据发布含新版 RGB JPEG 和关节修正，而官方指南固定的旧 RoboTwin 转换脚本直接使用 `cv2.imdecode`。
因此代码固定、ZIP 校验通过，不代表转换结果已经兼容；必须先做样本验收。下载工具不会修改第三方代码。

## 转换顺序

1. 下载并校验一个任务，例如 `adjust_bottle`。
2. 查看 ZIP 内目录，解压前检查条目路径，拒绝绝对路径、`..` 和越界软链接，不无条件解压未检查的 ZIP。
3. 确认该任务恰有 50 条示范，读取一条轨迹查看 HDF5 字段。
4. 根据图像编码版本选择匹配的 RGB 解码方式，保存相机样本确认颜色；避免旧解码器混用新 JPEG 导致红蓝通道交换。训练编码和推理相机也保持同一 RGB 语义。
5. 核对 14 维物理状态/动作：左臂 6、左夹爪 1、右臂 6、右夹爪 1；主相机和左右腕相机共三路。不要把末端位姿误作关节角。
6. 用选定转换器生成 LeRobot v2.1 或 v3.0，按一条示范、一个任务、全 50 任务的顺序扩大。
7. 检查图像、状态、动作、指令、未来帧对齐及 episode 边界。
8. 生成 `<robot_config_name> <LeRobot目录>` 形式的清单，路径建议不含空格；不能把 ZIP 或原始 HDF5 路径当作 LeRobot 输入。
9. 只用 clean 训练数据计算归一化统计；独立 robot 配置沿用官方关节/相机映射，指向 clean 统计，训练推理共用。不要套用上游 clean+randomized 统计。
10. 记录转换器版本、参数、示范数、字段和统计文件摘要。

清单示例（仅展示格式，当前不生成指向不存在目录的可训练清单）：

```text
robotwin_clean /absolute/ssd/path/lerobot/adjust_bottle
robotwin_clean /absolute/ssd/path/lerobot/beat_block_hammer
```

原始 ZIP 校验只能证明下载文件一致，不能证明数据转换或训练正确。
参考：[数据格式指南](../lingbotvla/data/vla_data/README.md)、[官方 RoboTwin 准备说明](../experiment/robotwin/README.md)。
