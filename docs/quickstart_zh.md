# 十分钟上手

## 安装

```bash
pip install qugrid            # 核心依赖只有 NumPy、SciPy、matplotlib、pandas,不需要任何量子 SDK
```

使用 [uv](https://docs.astral.sh/uv/) 则执行 `uv add qugrid`。可选生态通过 extras 安装:`qugrid[qiskit]`、`qugrid[dwave]`、`qugrid[pennylane]`、`qugrid[pandapower]`,或一次装全 `qugrid[all]`。

检查安装:

```bash
qugrid doctor
```

## 三行代码解决第一个问题

对 IEEE 9 节点系统做主动解列(controlled islanding):扰动后把电网分成两个自平衡的孤岛,尽量少切线路,同时让每个岛内的发电与负荷平衡。

```python
import qugrid as qg

result = qg.solve(qg.problems.Islanding(qg.cases.case9()), solver="qaoa", seed=0)
print(result.summary())
```

这三行做了四件事:把解列问题编码成 QUBO(每条母线一个二值变量);在内置的态矢量模拟器上运行 QAOA;把最优比特串解码回解列方案(岛集合、切除线路、功率不平衡量);并且因为问题足够小,枚举出精确最优解作为对照。

## 读结果

```python
result.decoded          # 工程答案:岛集合、切除线路、每个岛的功率不平衡(MW)
result.feasible         # 原始约束(惩罚项之前的约束)是否满足
result.gap()            # 与经典最优解的相对差距(0.0 = 达到最优)
result.success_probability()  # 单次测量得到最优态的概率
result.resources        # 量子比特数、运行时间、优化器迭代次数
```

这个库反复强调两个习惯:评价**解码后的工程答案**,而不是比特串;报告任何量子结果时,同一个 `Result` 里就带着经典参照,不要丢下它。

## 画图

```python
import matplotlib.pyplot as plt

qg.viz.use_style()
d = result.decoded
qg.viz.plot_network(qg.cases.case9(), islands=d["islands"], cut_edges=d["cut_lines"])
plt.savefig("islands.png", dpi=150, bbox_inches="tight")
```

## 同一问题,换求解器对比

```python
prob = qg.problems.Islanding(qg.cases.case9())
for s in ("exact", "sa", "qaoa"):
    r = qg.solve(prob, solver=s, seed=0)
    print(f"{s:6s} objective={r.objective:10.3f} gap={r.gap():.2e} feasible={r.feasible}")
```

同一个问题对象,三个求解器,一张对比表——这就是这个库的核心工作流。

## 一条命令跑完整演示

```bash
qugrid demo --figure islands.png
```

## 下一步

- 不了解量子计算 → 先读[给电力工程师的量子计算入门](primers/quantum-for-power.md)(英文,10 分钟),再做教程 01。
- 不了解电力系统 → 先读[给量子研究者的电力系统入门](primers/power-for-quantum.md),再做教程 02。
- 直接开始研究 → [示例库](zoo.md):十个带经典参照的完整研究脚本。

中文材料目前包括本页;教程与文档正文为英文。术语在两个 primer 中英对照给出,遇到不确定的概念,建议以英文原文为准。
