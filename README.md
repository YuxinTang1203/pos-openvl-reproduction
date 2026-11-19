# PoS + OpenVL on PyRUA-Lean

在 **PyRUA-Lean baseline** 上集成 PoS 状态维护与 OpenVL 视觉工具路由，复用 PyRUA 的代码执行、机器人动作接口和基础感知链路。

- [实现 Pipeline](PIPELINE.md)：数据流、模块职责、状态更新、完成审核与源码入口。
- [复现指南](REPRODUCTION.md)：环境要求、依赖安装、配置、任务运行与验收。
- [Codex 交接说明](CODEX_HANDOFF.md)：让另一个 Codex 按步骤复现。
- `source/`：PyRUA-Lean 源码、PoS/OpenVL 实现、测试及策略适配器。
- `scripts/`：环境检查、配置生成和服务启动工具。
- `requirements/`：分层依赖及已测试的软件环境快照。
- [Errors 与表现](logs/ERRORS-AND-PERFORMANCE.md)：按批次整理的成绩、故障分类和原始事件定位。
- [全部原始日志](logs/README.md)：20 个项目归档、44 条完整任务索引、辅助探针与图片，Release 下载。
- [验证记录](VALIDATION.md)：本地软件检查结果。

使用 `python scripts/configure.py` 生成 PoS + OpenVL 配置；使用 `--variant baseline` 生成同一 PyRUA 执行栈的 baseline 配置，以便对照。

无需训练。完整任务运行需要另行准备模型权重、RoboDojo/XPolicyLab 仿真环境与模型端点。开始前运行 `python3 scripts/verify_package.py` 校验文件。
