# RoboDojo PoS + OpenVL historical logs

这是一批真实 `insert_tubes` 运行的历史日志：2026-11-14 mainline 批次的 **全部四条 A 条件**，即原始 PoS + OpenVL fusion。A 是该批实验的 fusion 对照条件，**不是纯 PyRUA baseline**。按条件整体收录，成功、低分、失败及基础设施故障都保留。

| Episode | Layout | Repeat | 官方得分 / 100 | 完整成功 | 有基础设施问题 |
| --- | --- | --- | --- | --- | --- |
| 01-A-l0-r0 | 0 | 0 | 0 | 否 | 是 |
| 04-A-l54-r0 | 54 | 0 | 0 | 否 | 是 |
| 09-A-l54-r1 | 54 | 1 | 0 | 否 | 否 |
| 12-A-l0-r1 | 0 | 1 | 100 | 是 | 是 |

本批 1/4 完整成功，平均官方得分 25/100。这是少量历史观测，不能据此判断相对纯 PyRUA baseline 的提升。基础设施问题不一定是最终失败原因，详情以事件和原记录为准。

## 阅读顺序

1. `INDEX.json`：四条 episode 的入口、官方结果与运行指标。
2. `identity/<job>-config.json`、`-deployment-selected.json`、`-prompt.txt`：实际配置、模型参数与提示。
3. `completed/<job>/openai_events.jsonl`：可用的模型响应、工具调用、PoS 调度与维护事件。
4. `completed/<job>/workspace/cell_*.py`：实际生成并保存的 Python 单元。
5. `completed/<job>/calls.jsonl`、`PUBLIC-TOOL-FEEDBACK.json`：公开 API 调用和工具反馈。
6. `completed/<job>/delivered-images/`：已下载的部分交付图像，INDEX 保留来源。
7. `official/<job>-official.json`：原始官方评分结构；`score-audit.json` 核对评分与运行身份。

## 与当前代码的关系

四条 migration 与本仓库 fusion 使用同一配置条件，但不是当前提交重跑的结果。源文件核对结果为 **118 个相同、7 个不同、9 个未包含于当前仓库**；逐文件记录见 `source-identity.json`。变化包括 runtime、PoS、视觉/机器人接口等，不能宣称当前版本已复现这次成绩。

本目录没有纯 PyRUA baseline 的配套实测记录，不能组成当前 baseline-vs-fusion 的正式比较。

## 完整性与脱敏

这里保存的是已有本地归档，而不是完整远端镜像：不含完整 episode 视频、原始 HTTP 请求体或独立的全部辅助模型对话文件；图像为当时下载的子集。模型事件和工具记录按已有文件保留，未按成绩删减 episode。

`PROVENANCE.json` 为每个复制文件记录原始 SHA256、发布 SHA256 和脱敏计数。机器上的用户/项目路径和 IP 服务地址已匿名化；机器人坐标、动作参数、模型输出、失败事件和官方得分保留。原日志内部嵌入的历史哈希仍指向脱敏前原文件。整包完整性由仓库根目录的 SHA256SUMS.json 校验。
