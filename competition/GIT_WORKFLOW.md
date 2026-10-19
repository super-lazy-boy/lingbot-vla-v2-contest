# 比赛代码管理

新比赛仓库默认私有，公开分享按组委会规则处理。保留官方历史和许可证。
`origin` 指向自己的比赛仓库，`upstream` 指向 `https://github.com/Robbyant/lingbot-vla-v2.git`。
官方基础提交见 `sources.lock.json`，RoboTwin 固定提交由 Git 子模块记录。

## 日常开发

`main` 保存可回退的基线，后续修改用 `codex/<主题>` 分支，验证后合并：

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/first-baseline
```

获取代码后初始化指定子模块：

```bash
git submodule update --init third_party/RoboTwin
```

不需要递归安装全部旧策略的依赖。每次实验记录代码 commit、数据/model revision、配置、种子、有效批量、步数、显存、耗时、checkpoint 路径和评测结果。
权重放工作目录，Git 只记录路径约定和校验值；提交日志或环境记录前检查敏感信息。
重要阶段可打 `prep-2026-10-19`、后续 `baseline-v1` 等标签，未训练/未评测的版本不使用暗示已完成的标签。

## 同步上游

```bash
git fetch upstream
git switch -c codex/update-upstream
git merge upstream/main
```

检查差异，运行准备工具测试，并重新验证数据/GPU 兼容性后再合并。训练期间保持代码和子模块版本稳定。

## 文件边界

`.gitignore` 排除本机数据、权重、输出、缓存、环境配置和密钥；`.gitattributes` 固定 Python/Shell/配置/文档为 LF，便于 Windows 准备、Ubuntu 执行。
数据和模型不上传到比赛 Git 仓库或 Git LFS，由原始发布地址、固定版本和校验值管理。

## 可选的 GitHub 自动测试

`templates/preparation.workflow.yml` 是 Windows / Ubuntu 两个平台的准备工具测试模板，不安装 GPU 环境，不下载数据。
当前使用的既有登录凭据没有 `workflow` 权限，因此模板没有放入 `.github/workflows/`，不会擅自扩展账号权限。
以后具备相应权限时，可复制为 `.github/workflows/preparation.yml` 并提交启用；不影响当前代码、下载或目标机部署。
