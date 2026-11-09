# 实验二 · 搜索算法

本实验有四个问题：用 A* 找迷宫最短路径、计算加权地形的最低通行费、用 Alpha-Beta 求博弈树根值，以及结合 A* 和博弈搜索确定寻宝的最优首步。每题提供课程样例、独立核验和可复跑的对照实验。

[快速运行](#1-快速运行) · [输入与目录](#2-输入与目录) · [四题设计与结果](#3-四题设计与结果) · [复现图表与计时](#4-复现图表与计时) · [核验口径与限制](#5-核验口径与限制) · [交付内容](#6-交付内容)

## 1. 快速运行

算法、测试和计时脚本使用 **Python 3.10 或更高版本**的标准库。课程样例随仓库提供，运行时无需下载数据。实验在 Windows 11 上完成，基础批次使用 Python 3.13.7，补充批次使用 3.12.14。每批的环境和计时范围以该目录的 `environment.json` 为准。

以下命令适用于 Windows PowerShell。已下载仓库或解压提交包时，直接在含 `verify_all.py` 的目录运行后两条命令。若 Python 的启动命令是 `py` 或 `python3`，替换命令中的 `python` 即可。

```powershell
git clone https://github.com/chenyunshu1031/experiment2-search-algorithms.git
cd experiment2-search-algorithms
python --version
python verify_all.py
```

预期看到四行 `official sample passed`，分题通过 **7、12、17、21** 项测试，共 **57** 项，最后显示 `All passed.`。只跑样例可执行 `python verify_all.py --smoke-only`。完整核验会在 `verification/<时间戳>/verification.json` 保存输出、测试日志和源码哈希。 [完整核验记录](verification/20261109T035804598391Z/verification.json) · [样例命令与输出](verification/20261109T025803635497Z/sample_runs.txt)

逐题查看正式输出可运行：

```powershell
python question1/maze_astar.py --input question1/sample.json
python question2/weighted_astar.py --input question2/sample.json
python question3/game_tree.py --input question3/sample.json --algorithm minimax
python question3/game_tree.py --input question3/sample.json --algorithm alphabeta
python question4/treasure_game.py --input question4/sample.json --algorithm minimax
python question4/treasure_game.py --input question4/sample.json --algorithm alphabeta
```

| 题目 | 样例应得结果 | 独立核验 |
| --- | --- | --- |
| 迷宫寻路 | 路径含 8 个坐标、共 7 步。等长路径均可 | BFS |
| 加权地形 | 最低通行费为 `7` | Dijkstra |
| 博弈树 | 两种算法均返回 `5` | 完整自底向上回传 |
| 宝藏分配 | 6×6 距离矩阵。两种算法在 H=2 均选 T2、收益差 −1，在 H=4 均选 T3、收益差 3 | Dijkstra 核对矩阵，宝藏排列穷举核对博弈值 |

## 2. 输入与目录

各题的 `sample.json` 就是最小输入示例。自备文件放在任意可读目录，用 `--input <路径>` 指定。网格坐标统一用从 0 开始的 `[行, 列]`，网格应为矩形，`X` 表示障碍。

| 输入示例 | 字段及约定 | 输出 |
| --- | --- | --- |
| [迷宫](question1/sample.json) | `grid`、`start`、`end`。上下左右移动，每步费用 1 | 有序坐标序列。无解为 `[]`，加 `--format json` 可输出 JSON 数组 |
| [加权地形](question2/sample.json) | `grid`、`start`、`end`。按进入格收费，起点初始和进入 E 免费 | 最低整数费用。无解为 `-1` |
| [博弈树](question3/sample.json) | `tree_structure`。非叶节点有 `children`，叶子有 `value`。根为 MAX，角色逐层交替 | 根得分 |
| [宝藏分配](question4/sample.json) | `grid`、`a_start`、`b_start`、`treasures`。宝藏用 `name`、`position`、`value` 描述，特殊格进入费为 1 | JSON 含点标签、距离矩阵、最优目标及收益差。默认输出 H=2 和 H=4 |

```text
question1/                  迷宫 A*、同分顺序、隔墙下界
question2/                  加权 A*、地形下界、同终点批量查询
question3/                  Minimax、Alpha-Beta、顺序与结构缓存
question4/                  距离矩阵、寻宝博弈及寻路和后继构造优化
question*/sample.json       课程样例输入
question*/test_*.py         功能、边界条件及与参考算法的对照测试
question*/results/          固定输入、逐次计时、汇总、搜索轨迹
verification/               逐题输出与完整测试记录
report/figures_reproduced/   五幅报告图、对应数值、绘图环境
verify_all.py               四题统一核验
report_figures.py           从保存的数据重建报告图
requirements-report-figures.txt  可选绘图依赖
```

## 3. 四题设计与结果

### 题一 · 迷宫寻路

A* 用坐标表示状态，`g` 记录已走步数，`h` 估计剩余步数，堆优先取 `f=g+h` 最小的状态。发现更短走法才更新距离和前驱。终点有效出堆后回溯前驱得到路径。默认用曼哈顿距离，同 `f` 优先较大的 `g`。

固定 FIFO，样例的扩展数从零启发的 16 降到曼哈顿的 13，路径仍为 7 步。但在 31×31 对角空地上，最短路径前缀大量同为 `f=60`，两者用 FIFO 都扩展 960 格。保持曼哈顿、只改同分顺序后，扩展降到 60 格。另设隔墙缺口场景检验障碍感知下界，扩展 5151→300，但在 46 张图中有 41 张的计时中位数反而上升。 [题一设计、搜索轨迹和复跑命令](question1/README.md)

### 题二 · 加权地形

样例走 8 步，前 7 步各付 1，进入 E 付 0。离 E 一格时真实剩余费用为 0，因此普通曼哈顿可能高估。修正下界扣除免费一步，终点邻域下界再计入必经邻格的最低费用。固定较大 `g` 优先，样例从零启发的 14 次扩展降到 8 次。

另设高价列、高价圈和多起点查询，分别检查下界能否识别必经费用、建表能否回本。高价整列场景中，行列下界使扩展 3060→120，但只按行列取最小值会漏掉局部高价圈。在 61×61 随机权值图上，单个起点查询用终点邻域 A* 更快。64 个起点共用一个终点时，逐次 A*、共享预算 512 的反向下界和精确反向搜索分别用时 132.463、104.249、5.208 ms，均含路径恢复。 [题二公式、费用明细和复跑命令](question2/README.md)

### 题三 · Alpha-Beta 剪枝

Minimax 从叶子回传收益，MAX 取最大值、MIN 取最小值。Alpha-Beta 在 `α ≥ β` 时跳过不能改变祖先决策的分支。按课程要求默认从左到右。样例根值为 5，Minimax 访问 15 个节点，左序剪枝访问 14 个。

顺序对照固定输入、关闭缓存，比较左序、逆序、固定种子随机顺序和首叶探测。首叶探测将搜索阶段读取从 7 降到 6，但加上 14 次排序预读后共读 20 次。结构缓存另测重复和无重复树，并检查上下界能否安全复用。16 份重复子树的访问数 2950→250，含准备时间仅从 13.243 降至 11.276 ms。 [题三回传、顺序与缓存对照](question3/README.md)

### 题四 · 宝藏分配

先计算 A、B、T1–T4 的点对距离，再按双方位置、剩余宝藏位掩码、累计收益、当前玩家展开博弈。每次取宝的净收益为宝藏价值减路费。H=2 时 T2 的即时净收益较有利，H=4 时先拿 T3 会改变对手后两步的去向与路费。原题 H=4 下，Minimax、标签顺序 Alpha-Beta、收益排序 Alpha-Beta 分别访问 65、47、37 个节点。

优化分开计量：寻路阶段比较逐点和多目标搜索，博弈阶段分别开关延迟创建后继与收益表。延迟构造在固定顺序下保留相同访问数，但省去被剪分支的对象创建，原题后继对象 44→36。另分别改变一个地形格费用与 T3 的价值，观察最短路改道、收益差和最优首步何时改变。 [题四矩阵、决策推演、消融与复跑命令](question4/README.md)

## 4. 复现图表与计时

### 从保存的数据重建报告图

只跑算法或计时可跳过此处的安装。以下命令创建独立绘图环境。已安装绘图依赖时，可直接用当前 Python 运行最后一条命令。

```powershell
python -m venv .venv
$plot = ".\.venv\Scripts\python.exe"
& $plot -m pip install -r requirements-report-figures.txt
& $plot report_figures.py --output .venv/repro/figures
```

原图使用 Python 3.12.14、Pillow 13.0.0 与 matplotlib 3.11.2。需要匹配该批版本时，安装 [该批依赖锁定文件](report/figures_reproduced/requirements-lock.txt)。图形在不同平台可能因字体而略有差异，可比对同时输出的数值 JSON。重建脚本读取保存的数据，重跑计时请用下一组命令。 [绘图环境与数据指纹](report/figures_reproduced/environment.json)

| 报告图 | 用来观察什么 | 数值 |
| --- | --- | --- |
| [图 1 · 迷宫搜索范围](report/figures_reproduced/q1_equal_f.png) | 对角空地上的同 `f` 候选为何会拖慢 FIFO | [数据](report/figures_reproduced/q1_data.json) |
| [图 2 · 加权路径与下界](report/figures_reproduced/q2_required_evidence.png) | 走 8 步为何只付 7，普通距离在哪里高估 | [数据](report/figures_reproduced/q2_data.json) |
| [图 3 · 样例剪枝位置](report/figures_reproduced/q3_pruning.png) | 左序下哪个叶子被跳过，为什么根值不变 | [数据](report/figures_reproduced/q3_data.json) |
| [图 4 · 不同深度的首步收益](report/figures_reproduced/q4_depth_choices.png) | T2、T3 的保证值何时改变排序 | [数据](report/figures_reproduced/q4_data.json) |
| [图 5 · 地形与宝藏价值](report/figures_reproduced/q4_controlled_compact.png) | 只改变一项时，路费何时进入平台、首步何时切换 | [数据](report/figures_reproduced/q4_controls_data.json) |

### 从输入重跑主要对照

在根目录运行。先设置 `$run`，再执行所需题目的命令。同名输出目录可能已经存在，重跑时重新设置 `$run`。各题 README 还列出其他对照、输出位置和计时范围。

```powershell
$run = Join-Path ".venv/repro" (Get-Date -Format "yyyyMMdd-HHmmss-fff")

python question1/heuristic_study.py --repeats 7 --output "$run/q1-heuristics"
python question2/geometry_heuristic_study.py --repeats 9 --output "$run/q2-heuristics"
python question2/batch_reverse_study.py --repeats 7 --output "$run/q2-batch"
python question3/benchmark.py --repeats 8 --output "$run/q3-order"
python question3/audit_report_evidence.py --results "$run/q3-order" --study
python question3/structural_study.py --repeats 12 --output "$run/q3-cache"
python question4/benchmark.py --repeats 8 --output "$run/q4-game"
python question4/successor_cost_study.py --repeats 11 --batch 3 --output "$run/q4-construction"
python question4/heldout_successor_study.py --repeats 9 --batch 3 --output "$run/q4-new-maps"
python question4/heuristic_field_study.py --repeats 7
```

题四最后一个脚本在 `question4/results/heuristic_field/<时间戳>/` 创建新批次。如需从头计算图 5 的数据，在仓库副本中使用已安装绘图依赖的 Python 运行 `python question4/required_evidence.py --study`，再重画图。该命令会更新副本中对应的单变量结果文件。

## 5. 核验口径与限制

- **先验算答案，再比较开销。** 检查路径是否经过合法四邻、累计费用是否正确，以及博弈算法是否返回相同根值。等长路径或等值首步可能有多个，核对代价或收益及最优集合。
- **区分计数。** A* 的扩展数只计有效出堆后实际检查邻居的状态，终点出堆即结束，计为一次取出而非扩展。博弈访问数含根，开启缓存时访问数含命中，另外统计终端评价、预读和后继构造。
- **比较同一批次。** `cases.json` 记录输入及适用的生成种子，`raw_runs.csv` 保存逐次测量，`summary.csv` 给出配置、计数和汇总，`environment.json` 记录运行环境、配置或源码指纹。题三和题四的基本随机顺序对照用种子 42，顺序敏感性另测多个种子。计时包含哪些准备步骤以该批记录为准。
- **检查复现差异。** 优先比对答案、扩展数、访问数和配置。耗时受系统、硬件和负载影响。需要追查差异时用 `--stats` 查看标准错误上的统计，用 `--trace-output` 保存轨迹，并确认样例和配置与表中一致。记录轨迹会增加开销，计时用独立脚本。
- **失败情况。** 迷宫与加权地形无解分别返回 `[]` 和 `-1`。输入字段有误、坐标越界或寻宝关键点间不连通时，按入口报错定位。寻宝状态空间随宝藏数和深度增长，较大的拓展地图主要用来单测寻路或有限深度搜索。

## 6. 交付内容

仓库提供算法、样例、测试、复现命令、实验数据和报告图。课程提交包另附五页报告 PDF 和实际终端运行截图。样例答案以命令行输出为准，其中题四的 JSON 可直接核对距离矩阵、H=2、H=4 与最优收益差。
