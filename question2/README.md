# 题二 · 加权地形

## 任务与运行

在四邻网格上求最低通行费。输入 JSON 包含 `grid`、`start`、`end`。由 `u` 走到 `v` 支付 `v` 的费用，起点初始不收费，进入 E 免费。输出最低整数费用，无解返回 `-1`。拓展实验允许其他格为 0，因此不能假设每一步都收费。 [课程样例](sample.json) · [实现](weighted_astar.py)

以下命令在本目录执行：

```powershell
python weighted_astar.py --input sample.json --stats --details
python weighted_astar.py --input sample.json --heuristic zero --stats
python weighted_astar.py --input sample.json --heuristic corrected --tie-break larger-g --stats --trace-output trace.json
python -m unittest -v
```

## 样例与下界

样例路径为 `(0,0)→(1,0)→(2,0)→(3,0)→(3,1)→(3,2)→(4,2)→(4,3)→(4,4)`，累计费用为 `0→1→2→3→4→5→6→7→7`。从 `(4,3)` 到 E 还需走一步，实际费用为 0，直接把曼哈顿距离 1 当费用估计就高估了。拓展反例中，普通曼哈顿实际返回 7，Dijkstra 的最优值为 6。 [逐格费用](results/official_path_costs.csv) · [高估诊断](results/unsafe_diagnostics.csv) · [报告图 2](../report/figures_reproduced/q2_required_evidence.png)

令 `D` 为曼哈顿距离，`c` 为非终点通行格的最低进入费，`m` 为终点可通行四邻中的最低进入费。

| 命令行选项 | 剩余费用下界 |
| --- | --- |
| `zero` | 恒为 0，作为基线 |
| `corrected` | `c × max(D−1, 0)`。扣除进入 E 的免费一步 |
| `goal-aware` | `D≥2` 时为 `c × (D−2) + m`，否则为 0。距离至少两步时，必须先进入一个终点邻格 |
| `euclidean`、`chebyshev` | 将取整欧氏距离或切比雪夫距离代入修正式 |
| `axis` | 结合地形行列松弛，用必须跨越的行、列的最低费用给出下界 |
| `reverse` | 从 E 反向确定一部分格的真实距离。用 `--reverse-budget` 控制建表规模 |

样例中 `c=m=1`，两个修正式均给起点下界 7，且路径达到下界。固定同 `f` 时优先较大 `g`，零启发扩展 14 格，两个修正下界均扩展 8 格。53 张不同地图上的几何与地形对照固定同分规则，比较 9 个配置，每配置测 9 次，并用独立 Dijkstra 核验代价及下界。

## 对照结果

- **终点邻格贵。** 当邻格都收费 9，`h₂` 比 `h₁` 多计入必付的 8。较大 `g` 优先时扩展 6560→160，耗时中位数 69.267→5.470 ms。把 `h₂` 改配 FIFO 仍扩展 6560 格，大量候选同为 `f=167`。
- **高价列与高价圈。** 整列高价绕不开，行列下界使扩展 3060→120。高价圈外的廉价格仍可能成为某行或某列的最小值，下界就会漏掉穿圈费用。固定较大 `g`，封闭高价圈的反向 32、128、512 总耗时分别为 39.517、23.745、23.002 ms，零启发的 21.640 ms 仍更低。开低价口后三个预算均慢于终点邻域下界，反向 512 虽省 161 次正向扩展，总耗时仍从 6.084 增到 6.961 ms。
- **同终点批量查询。** 在 9 张地图上各比较 1、4、16、64 个起点，均计入准备和路径恢复。反向沿 `v→u` 时仍收原方向 `u→v` 的进入费。共享预算 512 的下界保留逐起点 A*，精确反向搜索则一次确定查询所需的距离和路径。单次短查询可能由 A* 更快。64 个起点时，精确反向法在 9 张图上均更快，也能一次判断多个起点不可达。

## 复跑与数据

```powershell
$run = Join-Path "../.venv/repro" (Get-Date -Format "yyyyMMdd-HHmmss-fff")
python geometry_heuristic_study.py --repeats 9 --output "$run/q2-heuristics"
python weighted_heuristic_study.py --repeats 8 --output "$run/q2-order-and-budget"
python course_axis_study.py --repeats 6 --output "$run/q2-axis"
python batch_reverse_study.py --repeats 7 --output "$run/q2-batch"
```

第一项固定同分策略比较下界。第二项另对照 FIFO、较大 `g` 与反向预算 32/128/512，54 个场景去重为 53 张地图。第三项单独比较行列松弛的建表与搜索成本。第四项固定地图和终点比较批量复用。批次目录含输入、逐次结果、汇总和计时范围。 [几何与地形下界批次](results/geometry_followup/20261105T094155277386Z) · [排序与反向预算批次](results/course_followup/20261105T030433002341Z) · [行列松弛批次](results/course_followup/20261104T065033472670Z) · [批量查询批次](results/course_followup/20261109T031416794508Z)


## 用旧图定规则，在新图检验

先用前一批 9 张地图、36 个查询批次选择阈值。候选为从 1、4、16、64 个起点开始切换到精确反向搜索，以及始终逐次 A*。每个旧批次按“选中方法耗时 ÷ 两个固定方法中较短的耗时”归一化，再对批次取平均。阈值 4 的结果为 1.153，略低于阈值 16 的 1.157，因此固定使用“少于 4 个起点用逐次 A*，其余用精确反向搜索”。

独立复测包含 6 张新地图。起点有集中在目标附近、分布于四个象限及部分被整列墙隔断三种布局，每类各两张图。每张图依次查询 1、4、16、64 个起点。两种固定方法和选择规则均预热，轮换顺序后各计时 7 次，共 504 次。计时包含选择开销、输入准备、搜索和路径恢复。每条路径、费用和不可达结果均用独立 Dijkstra 核验。

| 策略 | 24 个批次的中位耗时之和（ms） |
| --- | ---: |
| 始终逐次 A* | 233.956 |
| 始终精确反向搜索 | 45.688 |
| 按 4 个起点切换 | 42.561 |

用两种固定方法的批次中位数判定哪一种更快，规则选中 20/24 次。两张集中图的单次查询中，精确反向法更快。两张分散图查 4 个起点时，逐次 A* 更快。例如 45×45 分散图，逐次 A* 耗时 1.687 ms，规则选择反向后耗时 2.304 ms。查询数能作为粗略判断，起点布局仍会改变总开销。

```powershell
python batch_policy_study.py --repeats 7 --output "../.venv/repro/q2-policy"
```

[原始计时](results/batch_policy/20261109T053420608241Z/raw_runs.csv) · [各批次汇总](results/batch_policy/20261109T053420608241Z/summary.csv) · [新图输入与核验值](results/batch_policy/20261109T053420608241Z/cases.json) · [阈值候选与环境](results/batch_policy/20261109T053420608241Z/environment.json)
