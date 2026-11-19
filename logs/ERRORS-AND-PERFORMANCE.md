# Errors 与目前可核查的表现

## 先看结论

这些是已有历史运行归档，**不是当前 GitHub 提交新跑的评测**。最新已归档的常规 PoS/OpenVL 对照批次包括 V34：原 A 与改版 G 在两个布局上均为 0 分、各 0/2 完整成功。更早 mainline 的 A 条件为 0、0、0、100，1/4 完整成功，均分 25/100；不同批次不能只挑成功案例或合并成一个“当前成功率”。

A 在这些批次通常表示当时的 PoS+OpenVL fusion 对照，**不是纯 PyRUA baseline**。各批源码和配置不同；目前仍不能从这些归档得出相对纯 PyRUA 的稳定提升结论。当前源码与历史 mainline 的逐文件差异见 [source-identity.json](robodojo-fusion-20261114/source-identity.json)。

## 成绩：按批次和条件分开

以下 44 条均有 SUMMARY 条目、对应 completed 目录和已定位的官方结果。归档内还保留更早的官方判分和辅助探针，它们没有再次混入此表。`completed` 是策略结束状态，不代表官方完整成功。分数从各 episode 的原始官方结果读取；这些已索引条目按官方汇总 0—100 尺度展示，原始结构另有 0—1 的 details 字段，不能混用。

| 批次 | 条件 | Episodes | 原官方得分 | 完整成功 | 失败请求尝试数 |
| --- | --- | --- | --- | --- | --- |
| pos-openvl-api-guidance-v33 | G | 2 | 0, 0 | 0/2 | 5 |
| pos-openvl-checked-v23 | E | 2 | 40, 0 | 0/2 | 4 |
| pos-openvl-checked-v23 | F | 2 | 0, 0 | 0/2 | 2 |
| pos-openvl-evidence-control | Q | 2 | 0, 0 | 0/2 | 1 |
| pos-openvl-execution-boundary-v26 | S | 2 | 0, 0 | 0/2 | 5 |
| pos-openvl-grasp-branch-v31 | A | 2 | 20, 20 | 0/2 | 1 |
| pos-openvl-grasp-branch-v31 | P | 2 | 0, 0 | 0/2 | 6 |
| pos-openvl-grasp-branch-v31 | R | 2 | 0, 0 | 0/2 | 1 |
| pos-openvl-mainline | A | 4 | 0, 0, 0, 100 | 1/4 | 4 |
| pos-openvl-mainline | B | 4 | 0, 0, 0, 0 | 0/4 | 6 |
| pos-openvl-mainline | E | 4 | 0, 0, 0, 0 | 0/4 | 5 |
| pos-openvl-marked-relations-v25 | A | 2 | 40, 0 | 0/2 | 1 |
| pos-openvl-marked-relations-v25 | R | 2 | 0, 0 | 0/2 | 2 |
| pos-openvl-plan-requirements-v34 | A | 2 | 0, 0 | 0/2 | 4 |
| pos-openvl-plan-requirements-v34 | G | 2 | 0, 0 | 0/2 | 2 |
| pos-openvl-reminder-presentation-v28 | A | 2 | 0, 0 | 0/2 | 4 |
| pos-openvl-reminder-presentation-v28 | B | 2 | 0, 0 | 0/2 | 5 |
| pos-openvl-reminder-presentation-v28 | C | 2 | 0, 0 | 0/2 | 2 |
| vision-readiness-v34 | V | 2 | 0, 20 | 0/2 | 4 |

完整逐条数据见 [EPISODES.csv](catalog/EPISODES.csv) 和 [EPISODES.json](catalog/EPISODES.json)。同名条件在不同批次未必同实现，应读取压缩包内 manifest、migration 和 condition_configs。

## 错误如何分类

1. **请求/传输故障**：原事件中的 HTTP400、HTTP504、ReadTimeout 等。`Download multimodal file timed out` 是供应商返回的错误，不证明图片链接失效，也不证明视觉模型判断错误。
2. **预算截止**：`status=timeout` 或剩余预算不足造成的请求结束，应与独立的服务故障区分。不能将所有 ReadTimeout 都归因服务不稳定。
3. **动作/API 反馈**：零步 IK 拒绝、执行未到达、闭爪后 holding=False 等。这些是可定位的反馈；holding=False 可能是空抓，也可能出现在有意接触探测中，不能自动计为违规。
4. **任务进展/几何问题**：生成代码没有依据反馈改变后续动作、持物与目标关系缺乏可验证几何、释放后尚未稳定等。必须结合单元代码、前后图像和调用记录判断，不能仅从错误字符串推出物理原因。

### 已索引的请求失败事件

这里统计的是 **失败请求尝试**，不是独立 episode，也不是去重后的事故数量；一次请求可能重试多次。

| 原事件分类 | 次数 |
| --- | --- |
| http_400 | 6 |
| http_504 | 1 |
| timeout | 57 |

### 可直接定位的例子

- `pos-openvl-api-guidance-v33/01-G-l0-r0`：`http_400`，角色 `planner`；压缩包内 `run/completed/01-G-l0-r0/openai_events.jsonl` 第 350 行。
- `pos-openvl-checked-v23/03-F-l54-r0`：`http_400`，角色 `planner`；压缩包内 `run/completed/03-F-l54-r0/openai_events.jsonl` 第 369 行。
- `pos-openvl-grasp-branch-v31/05-R-l54-r0`：`http_400`，角色 `planner`；压缩包内 `run/completed/05-R-l54-r0/openai_events.jsonl` 第 417 行。
- `pos-openvl-plan-requirements-v34/01-A-l0-r0`：`http_504`，角色 `planner`；压缩包内 `run/completed/01-A-l0-r0/openai_events.jsonl` 第 66 行。
- `pos-openvl-plan-requirements-v34/02-G-l0-r0`：`http_400`，角色 `planner`；压缩包内 `run/completed/02-G-l0-r0/openai_events.jsonl` 第 721 行。
- `pos-openvl-reminder-presentation-v28/06-A-l54-r0`：`http_400`，角色 `planner`；压缩包内 `run/completed/06-A-l54-r0/openai_events.jsonl` 第 259 行。

逐条错误、工具调用与 `file + line` 定位见 [ERRORS.json](catalog/ERRORS.json)。归档保留所有低分、失败与中断记录，汇总本身不自动给每条任务指定唯一失败原因。

## 建议的查看顺序

先选上表某个批次/条件 → 在 EPISODES.csv 找到 job → 在 ERRORS.json 找失败事件 → 下载对应 Release ZIP → 打开指定行附近的模型事件、workspace/cell、calls 和前后图像 → 对照官方结果。辅助探针与服务诊断单独归档，不能当成完整机器人任务成绩。
