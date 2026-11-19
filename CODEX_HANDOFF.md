# Codex reproduction handoff

先读 README.md、PIPELINE.md、REPRODUCTION.md、MANIFEST.json 与 VALIDATION.md。目标是在 PyRUA-Lean baseline 上复现 PoS + OpenVL。

1. 运行 scripts/verify_package.py，核对 Python、GPU、benchmark、权重和模型端点。
2. 创建 Python 3.11 环境，按 requirements/cpu.txt 安装依赖与 source/，运行 scripts/smoke.sh。
3. 默认运行 scripts/configure.py，确认 kind=v17_ablation、profile=fusion、openvl.enabled=true。对照组用 --variant baseline，确认 kind=baseline。
4. 在 Linux NVIDIA 主机准备配套的 torch/torchvision、基础感知模型与 RoboDojo/XPolicyLab，依照指南检查服务。
5. configure.py --benchmark 复制到新的策略目录，拒绝覆盖现有策略。baseline 和 fusion 使用不同的策略目录与输出目录。
6. 两组沿用同一 PyRUA 执行栈、模型、动作接口、任务布局、预算与评分器，只改变 migration 配置。保存配置和版本信息。
7. 先验证软件和服务，再运行官方 episode；保留所有结果，按官方完整成功定义统计。不得用单元测试结果代替任务成绩。
8. 缺少 simulator、原始 layout、策略 wrapper、固定模型版本或输入帧时，列出准确缺项；区分运行链复现与历史实验结果复现。

使用目标机器实际已授权的 GPU。完整评测阶段会调用配置的模型端点；安装和软件测试不发起模型请求。无需训练。
