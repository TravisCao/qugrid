<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="QuGrid" src="docs/assets/logo.svg" width="620">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/TravisCao/qugrid/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/TravisCao/qugrid/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-2a78d6">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-1baf7a">
  <a href="https://traviscao.github.io/qugrid/"><img alt="Docs" src="https://img.shields.io/badge/docs-mkdocs--material-eb6834"></a>
</p>

**面向电力系统研究的量子计算库:从 MATPOWER 算例到量子算法,只需三行代码。**

```python
import qugrid as qg

result = qg.solve(qg.problems.Islanding(qg.cases.case9()), solver="qaoa", seed=0)
print(result.summary())
```

QuGrid 服务两类研究者:想研究量子算法、但**不想离开本领域的工具、单位和证据标准**的电力系统研究者;以及想要一批经得起电力工程师评审的电网问题的量子计算研究者。

[English](README.md) · [中文](README.zh.md) · [文档](https://traviscao.github.io/qugrid/) · [十分钟上手](https://traviscao.github.io/qugrid/quickstart_zh/)

---

## 为什么需要这个库

在 MATPOWER/pandapower 与 Qiskit/Ocean/PennyLane 之间存在一条断层,大量研究质量消失在这里:惩罚权重没有文档的手搓 QUBO 编码、用 Ising 能量而不是兆瓦报告的结果、悄悄略去本领域实际使用的经典求解器的对比实验。QuGrid 用三层结构填上这条断层,每一层都对自己做的事保持诚实:

<p align="center"><img alt="QuGrid 架构" src="docs/assets/architecture.svg" width="860"></p>

1. **电力问题层**使用工程单位。`UnitCommitment`(机组组合)、`Islanding`(主动解列)、`PMUPlacement`(PMU 最优配置)、`EconomicDispatchQUBO`(经济调度)、`dc_power_flow`(直流潮流)、N-1 安全校核数据集、风电场景生成——全部构建在 `Network` 类上:保留 MATPOWER 列语义、直接读取 MATPOWER `.m` 文件、支持 pandapower 转换、内置 7 个标准算例(PJM 5 节点到 IEEE 118 节点)。
2. **数学编码层**是精确、有测试的代数:约定固定的 `QUBO ⇄ Ising` 互转、带精确平方惩罚展开的 `QUBOBuilder`(用于你自己的新问题)、带 2 的幂填充与 Hermitian 扩张的 `LinearSystemProblem`。测试套件在 1e-9 精度上验证全部约定。
3. **求解器层**运行在纯 NumPy 态矢量内核上——QAOA、VQE、HHL、VQLS、量子核方法、量子玻尔兹曼机——**不依赖任何量子 SDK**;旁边就是每个结论都必须面对的经典基线:精确枚举、固定种子模拟退火、LU 分解、牛顿-拉夫逊。同一个问题对象可一键导出到 Qiskit、D-Wave Ocean、PennyLane,用于厂商工具链或真机。

所有求解器返回同一个 `Result` 对象:解码后的工程答案、**原始约束**(而非惩罚项代理)的可行性、同一次运行中算出的与经典参照的 `gap()`、真机实验将面对的成功概率、以及资源账单(量子比特数、电路评估次数、运行时间)。

## 六十秒证据

IEEE 9 节点系统的主动解列。QUBO 精确最优解切除 2 条线路,两个孤岛的功率不平衡为 +10.3 / −5.0 MW;深度 2 的 QAOA 找到同一方案:

<p align="center"><img alt="IEEE 9 节点系统解列结果" src="docs/assets/hero_islanding.png" width="560"></p>

用 HHL 解直流潮流,同时给出大多数论文不放在一起的两个数:误差,以及每一位精度在后选择概率上的代价:

<p align="center"><img alt="HHL 误差解剖" src="docs/assets/hero_hhl.png" width="720"></p>

以下数字来自自校验示例脚本(每个脚本对自己的结论写断言,结论失效即退出非零):

| 研究 | 量子结果 | 同一次运行中的经典参照 |
|---|---|---|
| 直流潮流,HHL @ 8 个时钟量子比特 | 相对误差 2.5e-3,最大相角误差 0.0067° | LU 解(精确) |
| 机组组合,2 机组 × 2 时段 | 精确 QUBO = 模拟退火 = \$2,908.00,离散化差距 \$0.00 | 机组组合枚举:\$2,908.00 |
| 主动解列,IEEE 9 节点 | 精确 = 模拟退火 = QAOA(p=2),gap 为 0 | 精确枚举 |
| PMU 配置,9 节点 / 14 节点 | 模拟退火找到 3 / 4 台,全网可观 | 精确最少台数:3 / 4 |
| 量子核方法,N-1 安全校核 | 带宽调好时测试精度 1.00,调错时 0.50 | RBF 核:1.00 |

## 安装

```bash
pip install qugrid          # 核心依赖只有 NumPy、SciPy、matplotlib、pandas,无量子 SDK
pip install "qugrid[all]"   # 加装 qiskit、dwave、pennylane、pandapower 适配器
qugrid demo                 # 30 秒端到端自检
```

## 选择你的入口

| 你是…… | 从这里开始 | 用时 |
|---|---|---|
| 电力研究者,不了解量子 | [十分钟上手](https://traviscao.github.io/qugrid/quickstart_zh/) → [写给电力工程师的量子入门](https://traviscao.github.io/qugrid/primers/quantum-for-power/) → 教程 01 | 40 分钟 |
| 量子研究者,不了解电网 | [写给量子研究者的电力入门](https://traviscao.github.io/qugrid/primers/power-for-quantum/) → 教程 02 | 40 分钟 |
| 直接开始做实验 | [速查表](https://traviscao.github.io/qugrid/primers/cheatsheet/) → [示例库](examples/) | 现在 |

五本已执行的教程 notebook 从零量子知识带到独立开展研究——`notebooks/01_hello_qugrid`(15 分钟)到 `05_qml_for_screening`(30 分钟),以文字讲解为主,每个术语先用电力系统的语言定义。[学习路径页](https://traviscao.github.io/qugrid/learning-paths/)按背景给出完整顺序。

中文材料:本页与[十分钟上手(中文)](https://traviscao.github.io/qugrid/quickstart_zh/);教程与文档正文为英文,术语在两个入门读物中给出中英对照。

## 示例库

[`examples/`](examples/) 中的十个单文件研究,风格取自 CleanRL:自包含、固定随机种子、笔记本电脑上几分钟跑完、自带断言校验、经典基线写在同一个文件里。复制一个、换上你的算例文件,就得到一篇论文实验部分的骨架。

| # | 研究 | # | 研究 |
|---|---|---|---|
| 01 | HHL 解直流潮流:误差解剖 | 06 | QAOA 深度研究(解列问题) |
| 02 | QAOA 解机组组合,诚实的成功概率 | 07 | 多种子求解器基准 → LaTeX 表格 |
| 03 | IEEE 9 节点主动解列 | 08 | 量子核 vs RBF(N-1 校核) |
| 04 | PMU 配置:松弛位不等式编码 | 09 | 量子玻尔兹曼机生成风电场景 |
| 05 | 离散化的代价:编码误差 vs 求解误差 | 10 | 混合牛顿-拉夫逊 + 变分线性求解器 |

## 诚实框

> **今天没有任何量子设备能在任何电力系统问题上胜过调优后的经典求解器,这个库永远不会暗示相反的结论。**直流潮流由稀疏 LU 在微秒级解决;国家级机组组合由混合整数规划每晚求解。QuGrid 让 2026 年真实存在的研究变得容易:编码代价、误差解剖、资源标度、离散化下的算法行为——经典基线永远在同一张表里。[诚实基准指南](https://traviscao.github.io/qugrid/honest-benchmarking/)是这段话的六条规则版本;API 默认强制执行其中大部分。

## 定位

- **Qiskit Optimization、OpenQAOA、D-Wave Ocean** 消费抽象 QUBO。QuGrid 负责它们之前的事(可信的电网问题构建、有文档的惩罚权重、离散化核算)和之后的事(解码回兆瓦、真实约束的可行性、领域标准基线)——并可一键导出到这三者。
- **MATPOWER 与 pandapower** 仍是电网数据和经典潮流的权威来源;QuGrid 读取它们的格式,不替代它们。
- **CleanRL 与 Tianshou** 启发了这个库的形态:小而全测试的内核,加上单文件、自校验的研究脚本。

## 引用

如果 QuGrid 支持了你的研究,请通过 [`CITATION.cff`](CITATION.cff) 引用(GitHub 的 "Cite this repository" 按钮)——并请引用各求解器 docstring 中标注的算法原始论文;HHL、QAOA、VQLS 与量子潮流文献是这些作者的贡献,不是本库的。

## 贡献

价值最高的贡献是来自你自己研究的问题构建:一个文件、一个测试,库里所有求解器、基准和绘图自动适用于它。见 [CONTRIBUTING.md](CONTRIBUTING.md) 与[问题构建提案模板](.github/ISSUE_TEMPLATE/formulation_proposal.yml)。

## 许可证

MIT。内置算例数据来自 [MATPOWER](https://matpower.org/) 测试算例(BSD 3-clause)。
