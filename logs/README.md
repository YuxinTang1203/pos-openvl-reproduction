# 原始运行日志与表现索引

- [先看 errors 和表现](ERRORS-AND-PERFORMANCE.md)
- [全部原始归档 Release](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/tag/historical-logs-2026-11-19)
- [逐条 episode CSV](catalog/EPISODES.csv)、[错误定位 JSON](catalog/ERRORS.json)、[下载索引与 SHA256](catalog/ARCHIVES.json)
- [可直接浏览的四条历史 fusion 样例](robodojo-fusion-20261114/README.md)

本次共 **20 个项目归档、25,855 个原文件**，原始数据 **2.91 GB**，ZIP 合计 **2.67 GB**（十进制）。原始文本、模型消息、工具反馈、生成代码、已有图片/视频等按项目归档，包含失败、中断与故障；本次凭据扫描没有命中，所收录原文件字节未改写。

| 项目/批次 | 原文件数 | 未压缩数据 | 下载 ZIP |
| --- | --- | --- | --- |
| pos-openvl-api-guidance-v33 | 1293 | 185.8 MB | [177.6 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-api-guidance-v33.zip) |
| pos-openvl-bounded-search-v27 | 7 | 0.1 MB | [0.0 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-bounded-search-v27.zip) |
| pos-openvl-checked-v23 | 236 | 11.2 MB | [1.3 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-checked-v23.zip) |
| pos-openvl-evidence-control | 860 | 98.3 MB | [91.8 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-evidence-control.zip) |
| pos-openvl-execution-boundary-v26 | 2137 | 206.0 MB | [179.0 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-execution-boundary-v26.zip) |
| pos-openvl-grasp-branch-v31 | 5167 | 707.2 MB | [676.8 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-grasp-branch-v31.zip) |
| pos-openvl-mainline | 967 | 80.8 MB | [56.2 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-mainline.zip) |
| pos-openvl-marked-relations-v25 | 3437 | 259.8 MB | [217.2 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-marked-relations-v25.zip) |
| pos-openvl-plan-requirements-v34 | 3445 | 451.5 MB | [430.0 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-plan-requirements-v34.zip) |
| pos-openvl-recovery-v22 | 4 | 0.0 MB | [0.0 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-recovery-v22.zip) |
| pos-openvl-reminder-presentation-v28 | 4651 | 649.1 MB | [624.3 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/pos-openvl-reminder-presentation-v28.zip) |
| api-reminder-causal-probe-v29 | 650 | 74.5 MB | [50.9 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/api-reminder-causal-probe-v29.zip) |
| api-reminder-boundary-probe-v30 | 452 | 21.7 MB | [14.3 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/api-reminder-boundary-probe-v30.zip) |
| api-contract-generalization-v32 | 60 | 1.0 MB | [0.4 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/api-contract-generalization-v32.zip) |
| api-contract-generalization-v32b | 304 | 1.6 MB | [0.6 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/api-contract-generalization-v32b.zip) |
| feedback-mechanism-study | 111 | 5.6 MB | [1.8 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/feedback-mechanism-study.zip) |
| vision-readiness-v34 | 2048 | 156.5 MB | [144.4 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/vision-readiness-v34.zip) |
| vision-layer-audit-20261116 | 21 | 0.4 MB | [0.1 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/vision-layer-audit-20261116.zip) |
| image-transport-20261118 | 4 | 0.0 MB | [0.0 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/image-transport-20261118.zip) |
| v17-eval | 1 | 0.0 MB | [0.0 MB](https://github.com/YuxinTang1203/pos-openvl-reproduction/releases/download/historical-logs-2026-11-19/v17-eval.zip) |

## 范围与完整性

这表示 **当前本地已归档、与 PoS/OpenVL 相关的可用原始数据**，不是远端所有历史数据的完整镜像。主线和实验变体、辅助生成探针、图像传输诊断均分开保留。独立几何模型分支不属于本次发布范围。

每个 ZIP 内有 ARCHIVE_MANIFEST.json，列出每个原文件的 SHA256、发布 SHA256、大小，以及省略项。依赖/源码备份、模型权重、凭据文件、已解包的重复压缩容器不作为运行日志重复上传；原容器的哈希与排除原因会保留。原有 raw HTTP 请求体或视频若本地没有，就不会被宣称已恢复。图像仅包含实际已归档部分。

目录中的 44 条完整任务索引使用 SUMMARY + completed + 官方结果进行定位；其他历史判分、局部轨迹、探针不重复算作这 44 条。不同版本/批次的成绩不合并成当前实现的成功率。仅小型索引和代表样例进入 Git，全部原日志与图像在 Release 中下载。
