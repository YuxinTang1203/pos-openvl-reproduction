# Reproduction introduction / guidelines

解压后先运行 `python3 scripts/verify_package.py` 校验交付文件。

实现结构、数据流和代码入口见 [PIPELINE.md](PIPELINE.md)。

## 1. 复现对象

本包在 **PyRUA-Lean baseline** 上复现 PoS 状态维护与执行循环、OpenVL 视觉工具路由，复用其代码执行、机器人动作接口和基础感知链路。默认配置来自现有 ablation_fusion.json：v17_ablation / fusion，关闭 static_rules、action_focus、reliability，保留 PoS 与 OpenVL。新增 configure.py 只把路径解析到当前解压目录，不改变核心算法。

源码整理自本地 PyRUA-Lean 集成快照。没有训练步骤。`--variant baseline` 使用 kind=baseline，`--variant fusion`（默认）使用 PoS + OpenVL；两组共用同一 PyRUA 执行栈。SAM3 按基础感知依赖安装，安装步骤见下文。

## 2. Requirements

| 层级 | 要求 |
| --- | --- |
| 软件回归 | Python >=3.10，推荐 3.11；CPU 即可；需要联网获取固定提交的 OpenVL；见 requirements/cpu.txt |
| 真实 SAM3 感知 | Linux、NVIDIA CUDA、匹配的 torch/torchvision、sam3 固定提交、SAM3 checkpoint；Mac CPU 测试不能替代这一步 |
| OpenVL | OpenVLHarness 固定提交 abd465d1b7271b024d94ff0f0ce8ee7c4242a79a |
| 完整任务 | 已安装的 RoboDojo + XPolicyLab、仿真器及场景/机器人资产、OpenAI-compatible 多模态模型端点、密钥、可用 GPU |
| 历史严格对齐 | 相同 benchmark/native wrapper、layout 0/54 文件与哈希、Qwen 端点版本、参数、输入和依赖环境 |

显存没有在任意新硬件上测得通用下限。旧适配器说明 SAM3 约需 8 GB；本包启动工具默认要求一张空闲卡至少 10,000 MiB 可用显存，这是准入设置，不是保证足够的最低硬件规格。仿真器还需额外显存。可用单独的 SAM3 卡和 policy/simulator 卡，分别指定 UUID。CPU 内存和磁盘按仿真器要求准备；分割模型权重约 3.4 GB，下载/安装还需临时空间。

上游 pyproject 的主 dependencies 为空，因此不能只执行 pip install 然后假设运行依赖齐全。本包提供分层依赖；**没有完整原环境 lockfile**，torch/CUDA/仿真器版本须在目标机器核对并在运行后冻结。

## 3. CPU 安装与软件验证

所有命令在解压后的包根目录执行，以下 python3.11 可换成已有 Python >=3.10：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/cpu.txt
python -m pip install -e ./source
python scripts/check_environment.py
bash scripts/smoke.sh
python scripts/configure.py
```

CPU 测试也会导入 OpenVL 可视化所需的 torch/torchvision，但不需要 CUDA。Linux 机器若要继续 GPU 阶段，应先按第 4 节安装配套的 CUDA torch/torchvision，再装 cpu.txt；已有版本会被保留。

测试使用合成环境/假分割器，检查 harness、路由、状态和分割传输契约；不会加载权重、调用付费模型或运行真实机器人。实际本机执行结果见 VALIDATION.md。已通过测试的 macOS / Python 3.11 依赖快照见 requirements/cpu-tested-macos311.txt；它不是 Linux/CUDA 锁文件，不应替代 GPU 环境约束。

## 4. GPU 环境、SAM3 与 OpenVL

先按目标机器驱动安装配套的 CUDA torch 与 torchvision，并验证：

```bash
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available()'
python -m pip freeze > configs/torch-before.txt
```

保留已装 torch/torchvision 的约束，避免 SAM3 安装时悄悄替换：

```bash
python - <<'PYTHON'
import importlib.metadata as m
from pathlib import Path
p=Path('configs/torch-constraints.txt')
p.write_text(''.join(f'{n}=={m.version(n)}\n' for n in ['torch','torchvision']) + Path('requirements/constraints.txt').read_text())
PYTHON
python -m pip install -c configs/torch-constraints.txt -r requirements/gpu.txt
python -m pip install -c configs/torch-constraints.txt 'sam3 @ git+https://github.com/RLinf/sam3.git@c1f6b1e83967c912f675d8975d437ccdc41e17d8'
python -m pip install -c configs/torch-constraints.txt 'openvlharness @ git+https://github.com/uclanlp/OpenVLHarness.git@abd465d1b7271b024d94ff0f0ce8ee7c4242a79a'
```

SAM3 权重从 facebook/sam3 获取。按其访问要求登录/取得模型访问权限后执行；本包未附带权重或访问凭据：

```bash
hf download facebook/sam3 sam3.pt --local-dir assets
export SAM3_CHECKPOINT_PATH="$PWD/assets/sam3.pt"
python scripts/check_environment.py --gpu
nvidia-smi -L
```

历史 SAM3 checkpoint 的 revision 与 SHA256 在现有记录中未完整固定；这意味着新下载权重只能用于运行链复现，严格结果复现前需从原服务器补齐并核验。

启动感知服务（终端 A，选择自己的空闲 GPU）：

```bash
python scripts/start_sam3.py --gpu-uuid GPU-YOUR-SAM3-UUID --port 8765
```

终端 B：

```bash
source .venv/bin/activate
export SAM3_CHECKPOINT_PATH="$PWD/assets/sam3.pt"
export PYRUALEAN_SAM3_URL=http://127.0.0.1:8765
curl --fail http://127.0.0.1:8765/health
```

该进程共享 segment 与 segment_all 的推理锁，使用无损 RGB、打包 mask 和请求身份校验。health ready 只说明 SAM3 服务已加载；仍需真实图像/任务验证分割质量。不需要启动 capabilities_visual.yml 内的占位 localhost 模型服务：此配置的 native 路由会替换后端，详见 openvl_routed.py。

## 5. 接入 RoboDojo / XPolicyLab 并运行

本包包含策略适配器，不包含完整 simulator、XPolicyLab 服务、场景资产或官方 layouts。先按你已有的 RoboDojo/XPolicyLab 安装流程准备它们。在 policy 环境安装目标 benchmark 的 XPolicyLab：

```bash
export BENCHMARK_ROOT=/absolute/path/to/your/robodojo
python -m pip install -e "$BENCHMARK_ROOT/XPolicyLab"
python scripts/configure.py --benchmark "$BENCHMARK_ROOT"
```

脚本复制到 XPolicyLab/policy/PyRUA_Reproduction，加入 migration 并把输出放在本包 outputs/fusion；目录已存在就拒绝覆盖。benchmark client 的 conda 环境需要其自带的 simulator 依赖。策略接口详情见 source/integrations/xpolicylab/PyRUA_Lean/README.md。

配置模型（占位符必须换成真实值；不要把密钥写进归档）：

```bash
export OPENAI_BASE_URL=https://YOUR-ENDPOINT/v1
export PYRUALEAN_BASE_URL="$OPENAI_BASE_URL"
export OPENAI_API_KEY=YOUR-KEY
export PYRUALEAN_API_KEY_ENV=OPENAI_API_KEY
export PYRUALEAN_MODEL=qwen3.8-flash
export PYRUA_ADMITTED_GPU_UUID=GPU-YOUR-POLICY-UUID
```

运行原有适配器 eval.sh，传入实际 policy 环境路径和 simulator conda 环境名。policy GPU 参数用与准入变量相同的 UUID；SAM3 分割在终端 A 的卡上运行：

```bash
bash "$BENCHMARK_ROOT/XPolicyLab/policy/PyRUA_Reproduction/eval.sh" \
  RoboDojo insert_tubes no-checkpoint arx_x5 joint 0 \
  "$PYRUA_ADMITTED_GPU_UUID" YOUR-SIMULATOR-GPU-ID \
  "$PWD/.venv" YOUR-SIMULATOR-CONDA-ENV
```

该命令是**通用任务运行**，由你的 benchmark 决定 layout 调度，不能据此声称只跑了历史 layout 0/54。基础策略预算为 max_turns=100、budget_s=5400、max_tokens=8192、reasoning_effort=high。端点必须支持图像及这些参数；如果换模型或调整参数，请作为新实验记录。

要精确复现历史 layouts 0/54，需先取得原始 layout 文件、可信 benchmark/wrapper 与 manifest，再按官方机制选择布局并核验哈希。

对照 PyRUA baseline：

```bash
python scripts/configure.py --variant baseline --benchmark "$BENCHMARK_ROOT"
```

这会创建 PyRUA_Baseline 策略，配置 kind=baseline。运行该目录下的 eval.sh，其他参数与 fusion 保持相同。两组应使用同一模型、布局、动作接口、预算和官方评分器，输出分别保存在 outputs/baseline 与 outputs/fusion。

## 6. 验收与复现记录

分别记录：软件测试结果；SAM3 实际推理与源图对应；policy 是否收到当前图像并产出工具调用；官方 episode 是否结束及其原始得分。保存 pip freeze、Python/torch/CUDA/GPU 信息、resolved config、layout/benchmark 哈希、模型名与端点版本（隐藏密钥）、完整工具轨迹和官方 _result.json。成功率只能按官方完整成功定义统计。

常见问题：Python 3.9 不支持本项目；mcp 必须 1.x；NumPy 保留 1.26 系列；torch/torchvision 要匹配；GPU 检查失败应换到已授权可用卡，不要绕过所有权检查。更换解压目录后重新运行 configure.py 生成绝对路径。


## 7. 已归档的真实任务日志

见 [logs/robodojo-fusion-20261114](logs/robodojo-fusion-20261114/README.md)。其中四条 A 条件均为 PoS + OpenVL fusion，不能误当纯 PyRUA baseline。保留全部成功/失败结果、故障事件、可用模型/工具轨迹及来源索引；日志与当前源码存在已记录差异，不是本次软件交付中新跑的任务。

全部相关历史归档与缺项见 [logs/README.md](logs/README.md)，错误和表现见 [logs/ERRORS-AND-PERFORMANCE.md](logs/ERRORS-AND-PERFORMANCE.md)。
