# 实验二 搜索算法

四道题从最短路和博弈树出发，最后把两者用在同一个寻宝问题里。题一用 A* 找最短路径，题二改为计算加权地形的最小通行费，题三比较 Minimax 和 Alpha-Beta，题四先用 A* 计算关键点之间的路费，再让两位玩家轮流选择宝藏。

## 1. 下载和运行

在 Windows PowerShell 中执行。已有本地副本或解压后的提交包时，直接进入含 `verify_all.py` 的目录，从第三行开始。核心算法、测试与计时只用 Python 3.10 或更新版本的标准库。保存的实验在 Windows 11 下运行，Python 版本以每批的 `environment.json` 为准。

```powershell
git clone https://github.com/chenyunshu1031/experiment2-search-algorithms.git
cd experiment2-search-algorithms
$py = "python"
& $py --version
& $py verify_all.py
```

若 Windows 只识别 `py`，将第三行改为 `$py = "py"`。统一入口会运行四道题的官方样例与 **57 项测试**，分题为 **7、12、17、21** 项。每次运行在 `verification/` 下生成带时间戳的 JSON，记录实际输出、测试日志和源码哈希。只查官方样例可加 `--smoke-only`。 [验收用测试日志](verification/20261109T035804598391Z/verification.json) · [样例命令与输出](verification/20261109T025803635497Z/sample_runs.txt)

## 2. 逐题查看结果

从仓库根目录执行以下命令。第四题的每条命令都运行 H=2 和 H=4。

```powershell
& $py question1/maze_astar.py --input question1/sample.json
& $py question2/weighted_astar.py --input question2/sample.json
& $py question3/game_tree.py --input question3/sample.json --algorithm minimax
& $py question3/game_tree.py --input question3/sample.json --algorithm alphabeta
& $py question4/treasure_game.py --input question4/sample.json --algorithm minimax
& $py question4/treasure_game.py --input question4/sample.json --algorithm alphabeta
```

| 题目 | 输入与预期输出 |
| --- | --- |
| [1. 迷宫寻路](question1/README.md) | JSON 含 `grid`、`start`、`end`。坐标为从 0 开始的行、列。样例从 `(0,0)` 到 `(3,4)`，输出 8 个坐标，共 7 步。无路时返回 `[]`。 |
| [2. 加权地形](question2/README.md) | 字段同题一，进入格子时收费，最后进入 E 免费。样例输出整数 `7`，无路时返回 `-1`。 |
| [3. 博弈树](question3/README.md) | JSON 的 `tree_structure` 存放树。根为 MAX，按输入顺序从左到右。Minimax 和 Alpha-Beta 均输出 `5`。 |
| [4. 宝藏分配](question4/README.md) | JSON 含 `grid`、`a_start`、`b_start`、`treasures`。输出标签、6×6 距离矩阵和两个深度的最优决策。两种算法均得 H=2 选 T2、收益差 −1，H=4 选 T3、收益差 3。 |

每个入口支持 `--help` 查看可选参数。`--stats` 将统计写到标准错误，正常答案仍从标准输出读取。把 `--input` 换成自备 JSON 即可运行其他输入，字段示例见各题的 `sample.json`。第四题遇到不可达的关键点会报错，因为原题没有定义跳过不可达宝藏的规则。

## 3. 目录结构

```text
question1/              maze_astar.py、迷宫启发函数与独立 BFS 核验
question2/              weighted_astar.py、加权启发函数与批量反向搜索
question3/              game_tree.py、Minimax、Alpha-Beta 与结构缓存
question4/              treasure_game.py、A* 距离矩阵与寻宝博弈
question*/sample.json   课程样例输入
question*/test_*.py     功能、边界和参考算法对照测试
question*/results/     已保存的输入、轨迹、计时与汇总
report/figures_reproduced/  五幅报告图及对应数值
verification/           样例输出与完整测试日志
verify_all.py           四题统一核验入口
report_figures.py       五幅报告图的重建入口
```

## 4. 复现报告图和对照实验

### 先用保存的数据重画报告图

绘图需要 Pillow 和 matplotlib，复现时使用 Pillow 13.0.0、matplotlib 3.11.2。建议先用 `& $py -m venv .venv` 创建虚拟环境，再将 `$py` 设为 `".\.venv\Scripts\python.exe"`。下列命令将五幅图和数值 JSON 写入 `.venv/repro/figures/`。图 1 是迷宫扩展范围，图 2 是路径费用与启发下界，图 3 是剪枝位置，图 4 是不同深度的首步收益，图 5 是单格费用与宝藏价值的单变量对照。 [原图及数据](report/figures_reproduced)

```powershell
& $py -m pip install -r requirements-report-figures.txt
& $py report_figures.py --output .venv/repro/figures
```

### 再从输入重跑对照

下列命令沿用报告里的重复次数。每次用新的 `$run`，结果和随仓库提供的计时批次分开保存。

```powershell
$run = Join-Path ".venv/repro" (Get-Date -Format "yyyyMMdd-HHmmss-fff")

& $py question1/heuristic_study.py --repeats 7 --output "$run/q1-heuristics"
& $py question2/geometry_heuristic_study.py --repeats 9 --output "$run/q2-heuristics"
& $py question2/batch_reverse_study.py --repeats 7 --output "$run/q2-batch"
& $py question3/benchmark.py --repeats 8 --output "$run/q3-order"
& $py question3/audit_report_evidence.py --results "$run/q3-order" --study
& $py question3/structural_study.py --repeats 12 --output "$run/q3-cache"
& $py question4/benchmark.py --repeats 8 --output "$run/q4-game"
& $py question4/successor_cost_study.py --repeats 11 --batch 3 --output "$run/q4-construction"
& $py question4/heldout_successor_study.py --repeats 9 --batch 3 --output "$run/q4-new-maps"
& $py question4/heuristic_field_study.py --repeats 7
```

题四启发值场脚本会在 `question4/results/heuristic_field/` 新建时间戳目录。要重新计算图 5 的单变量数据，可在仓库副本中运行 `& $py question4/required_evidence.py --study`。它会在副本内更新 `controlled_parameter_results.csv` 和 `controlled_parameter_oracles.json`，随后运行绘图命令即可重画图 5。

| 报告中的对照 | 看哪份数据 |
| --- | --- |
| 题一启发函数、同分顺序及隔墙下界 | [扩展数与耗时](question1/results/course_followup/20261105T024931827094Z/summary.csv) |
| 题二几何与地形下界 | [扩展数与含建表的耗时](question2/results/geometry_followup/20261105T094155277386Z/summary.csv) |
| 题二同终点批量查询 | [三种方法的 756 次计时汇总](question2/results/course_followup/20261109T031416794508Z/summary.csv) |
| 题三访问顺序和结构缓存 | [顺序对照](question3/results/report_followup/20261105T2245261782216/summary.csv) · [结构缓存](question3/results/structural_followup/20261105T2239574201109/summary.csv) |
| 题四距离、深度及首步变化 | [距离矩阵](question4/results/distance_matrix.csv) · [博弈开销](question4/results/game_summary.csv) · [单变量结果](question4/results/controlled_parameter_results.csv) |
| 题四启发值场与延迟构造 | [启发值场](question4/results/heuristic_field/20261102T105518857084Z/summary.csv) · [构造与收益表](question4/results/successor_cost/20261106T021641697877Z/summary.csv) · [新地图复测](question4/results/heldout_successor/20261109T021529689579Z/summary.csv) |

核对复跑结果时，先看答案和固定配置下的计数，再看耗时。以同一批次的中位数比较性能，并检查计时是否包含建表、排序或路径恢复。相同输入和顺序的扩展数可逐项比对，耗时会随机器和系统负载变化。
