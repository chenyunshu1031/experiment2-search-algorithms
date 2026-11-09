# 实验二 · 搜索算法

本实验有四个问题：用 A* 找迷宫最短路径、计算加权地形的最低通行费、用 Alpha-Beta 求博弈树根值，以及结合 A* 和博弈搜索确定寻宝的最优首步。每题提供课程样例、独立核验和可复跑的对照实验。

[环境与依赖](#1-环境与依赖) · [输入与目录](#2-数据与目录) · [运行顺序](#3-运行顺序与预期输出) · [实验配置](#4-实验配置与主要结果) · [结果与提交](#5-结果与提交) · [复现与限制](#6-复现与限制)

## 1. 环境与依赖

| 用途 | Python 与依赖 | 验证环境 |
| --- | --- | --- |
| 求解、测试、性能对照 | Python 3.10+，使用标准库 | Windows 11。基础批次 Python 3.13.7，补充批次 Python 3.12.14 |
| 重建报告图及单变量配图 | 见 [绘图依赖](requirements-report-figures.txt)，原批次为 Python 3.12.14、Pillow 13.0.0、matplotlib 3.11.2 | [锁定版本](report/figures_reproduced/requirements-lock.txt) · [绘图环境](report/figures_reproduced/environment.json) |

Windows PowerShell 中可用下列命令取得仓库。通过 GitHub 下载代码 ZIP 时，解压后直接进入含 `verify_all.py` 的目录即可。核心求解可以在其他支持相应 Python 版本的平台运行，本项目的实测环境为 Windows 11。

```powershell
git clone https://github.com/chenyunshu1031/experiment2-search-algorithms.git
cd experiment2-search-algorithms
python --version
```

下面的流程以 `python` 为启动命令。若机器上使用的是 `py` 或 `python3`，请替换命令前缀。运行算法和计时可直接进入第 3 节。绘图单独安装：

```powershell
python -m venv .venv
$plot = ".\.venv\Scripts\python.exe"
& $plot -m pip install -r requirements-report-figures.txt
```

macOS 或 Linux 上的虚拟环境解释器为 `.venv/bin/python`。需要对齐原图的依赖版本时，在 Python 3.12 环境中将安装命令的文件替换为 `report/figures_reproduced/requirements-lock.txt`。

## 2. 数据与目录

各题的 `sample.json` 提供可直接运行的课程输入。自备文件使用 UTF-8 JSON，放在任意可读目录，用 `--input "文件路径"` 指定。网格坐标统一用从 0 开始的 `[行, 列]`，行列数一致且网格非空，`X` 表示障碍。

| 输入示例 | 字段及约定 | 输出 |
| --- | --- | --- |
| [迷宫](question1/sample.json) | `grid`、`start`、`end`。元素为 `S`、`E`、`.` 或 `X`。上下左右移动，每步费用 1 | 有序坐标序列。无解为 `[]`，加 `--format json` 可输出 JSON 数组 |
| [加权地形](question2/sample.json) | `grid`、`start`、`end`。地形为非负整数、数字字符串或 `S/E/X`。按进入格收费，起点初始和进入 E 免费 | 最低整数费用。无解为 `-1` |
| [博弈树](question3/sample.json) | `tree_structure`。非叶节点有 `children`，叶子有 `value`。叶值为有限实数。根为 MAX，角色逐层交替，可省略玩家标记 | 根得分 |
| [宝藏分配](question4/sample.json) | `grid`、`a_start`、`b_start`、`treasures`。宝藏用 `name`、`position`、`value` 描述，普通地形使用正整数费用，玩家与宝藏位置互异，进入这些特殊格付 1 | JSON 含点标签、距离矩阵、最优目标及收益差。默认输出 H=2 和 H=4 |

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

## 3. 运行顺序与预期输出

四题算法直接求解 JSON 输入。先用课程样例检查入口与输入，接着做完整核验，再运行指定求解配置或计时对照。自定义输入可用对应的 `--input` 命令求解，入口会校验字段和取值。

```powershell
python verify_all.py --smoke-only
python verify_all.py
```

完整核验应显示四行 `official sample passed`，各题通过 **7、12、17、21** 项测试，共 **57** 项，最后显示 `All passed.`。两种核验均写出 `verification/<时间戳>/verification.json`，其中 `questions[].official_output` 是四题样例答案，完整运行还记录测试日志和源码哈希。 [完整核验记录](verification/20261109T053643355333Z/verification.json) · [样例命令与输出](verification/20261109T025803635497Z/sample_runs.txt)

[终端运行截图](verification/20261109T030652909457Z/terminal_verification.png)记录一次较早的四题核验，共 56 项测试，可与[同批 JSON 日志](verification/20261109T030652909457Z/verification.json)对照。当前 57 项测试的结果见上方完整核验记录。

逐题运行 Minimax、Alpha-Beta 和两种 A*，可用以下命令：

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

### 保存与读取答案

题一使用 `--format json` 可把 Python 元组序列改为 JSON 坐标数组。题二与题三直接输出数字。题四用 `labels` 标明矩阵行列，`H=2` 与 `H=4` 各含 `best_target` 和 `best_value`。只查看矩阵可用 `--matrix-only`，指定深度用 `--depth`。统一的样例 JSON 已由上面的核验命令保存。

```powershell
python question1/maze_astar.py --input question1/sample.json --format json
python question4/treasure_game.py --input question4/sample.json --matrix-only
python question4/treasure_game.py --input question4/sample.json --depth 4 --stats
```

`--stats` 将统计写到标准错误，答案仍从标准输出读取。`--trace-output` 另存搜索过程，适合排查路径、剪枝和缓存事件。

## 4. 实验配置与主要结果

### 输入分组与随机性

课程样例用于核对题意和输出，自设场景用于回答具体问题，另选地图用于复测延迟构造。搜索实验直接在给定状态空间上求解，因此这里按输入用途分组。各对照在同一输入上配对运行，保存输入和种子便于复跑。

| 输入组 | 配置、种子与数据 |
| --- | --- |
| 题一地图 | 46 张，含空地、走廊、无解、隔墙与随机图。各地图的生成种子随 [输入清单](question1/results/course_followup/20261105T024931827094Z/cases.json) 保存 |
| 题二加权地图 | 几何与地形下界对照为 53 张不同地图、9 个配置。[地图与配置](question2/results/geometry_followup/20261105T094155277386Z/cases.json)。批量对照用种子 `20261109`，每图查询 1、4、16、64 个起点，共 9 张图，预算为 512。[批量输入](question2/results/course_followup/20261109T031416794508Z/cases.json) |
| 题三博弈树 | 38 个输入，每个配置测 8 次，基础随机顺序种子 42。顺序敏感性另在 4 个输入上用 0–23 共 24 个顺序种子，每种重复 3 次。结构缓存另比较重复与无重复子树。[基础输入](question3/results/report_followup/20261105T2245261782216/cases.json) · [逐种子结果](question3/results/report_followup/20261105T2245261782216/order_study_by_seed.csv) |
| 题四寻宝 | 正式样例比较 H=2 与 H=4，基础随机排序种子 42。延迟构造另在 8 张图上复测，每图两种深度，地图种子为 `202611091`–`202611094`。[基础输入](question4/results/cases.json) · [新图输入](question4/results/heldout_successor/20261109T021529689579Z/cases.json) |

### 默认配置与对照变量

| 题目 | 正式默认 | 对照中改变的变量 |
| --- | --- | --- |
| 迷宫 | A*，`manhattan`，同 `f` 按 `larger-g` | 固定 FIFO 比较零启发、三种几何距离；固定曼哈顿比较同分规则；隔墙实验另测 `barrier` |
| 加权地形 | A*，`goal-aware`，同 `f` 按 `larger-g` | 固定同分规则比较几何下界、行列松弛与反向预算 32、128、512。另固定下界比较 FIFO 与较大 `g` |
| 博弈树 | `alphabeta`，`left`，缓存关闭 | 与完整 Minimax 对照，再比较逆序、随机、首叶探测、身份缓存与结构缓存 |
| 宝藏分配 | 有向点对 A*，`alphabeta`、标签顺序，H=2 和 H=4，缓存、收益表、延迟构造均关闭 | 固定矩阵改变顺序和缓存；后继构造和收益表分开开关；寻路阶段另比较批量和估值方式；地形费用、T3 价值各做单变量对照 |

### 题一 · 迷宫寻路

A* 用坐标表示状态，`g` 记录已走步数，`h` 估计剩余步数，堆优先取 `f=g+h` 最小的状态。发现更短走法才更新距离和前驱。终点有效出堆后回溯前驱得到路径。默认用曼哈顿距离，同 `f` 优先较大的 `g`。

固定 FIFO，样例的扩展数从零启发的 16 降到曼哈顿的 13，路径仍为 7 步。但在 31×31 对角空地上，最短路径前缀大量同为 `f=60`，两者用 FIFO 都扩展 960 格。保持曼哈顿、只改同分顺序后，扩展降到 60 格。另设隔墙缺口场景检验障碍感知下界，扩展 5151→300，但在 46 张图中有 41 张的计时中位数反而上升。 [题一设计、搜索轨迹和复跑命令](question1/README.md)

### 题二 · 加权地形

样例走 8 步，前 7 步各付 1，进入 E 付 0。离 E 一格时真实剩余费用为 0，因此普通曼哈顿可能高估。修正下界扣除免费一步，终点邻域下界再计入必经邻格的最低费用。固定较大 `g` 优先，样例从零启发的 14 次扩展降到 8 次。

另设高价列、高价圈和多起点查询，分别检查下界能否识别必经费用、建表能否回本。高价整列场景中，行列下界使扩展 3060→120，但只按行列取最小值会漏掉局部高价圈。在 61×61 随机权值图上，单个起点查询用终点邻域 A* 更快。64 个起点共用一个终点时，逐次 A*、共享预算 512 的反向下界和精确反向搜索分别用时 132.463、104.249、5.208 ms，均含路径恢复。 另用旧图确定 4 个起点的切换阈值，在 6 张新图上独立检验。24 个批次中有 20 次选中中位耗时更低的固定方法。 [新图复测汇总](question2/results/batch_policy/20261109T053420608241Z/summary.csv) [题二公式、费用明细和复跑命令](question2/README.md)

### 题三 · Alpha-Beta 剪枝

Minimax 从叶子回传收益，MAX 取最大值、MIN 取最小值。Alpha-Beta 在 `α ≥ β` 时跳过不能改变祖先决策的分支。按课程要求默认从左到右。样例根值为 5，Minimax 访问 15 个节点，左序剪枝访问 14 个。

顺序对照固定输入、关闭缓存，比较左序、逆序、固定种子随机顺序和首叶探测。首叶探测将搜索阶段读取从 7 降到 6，但加上 14 次排序预读后共读 20 次。结构缓存另测重复和无重复树，并检查上下界能否安全复用。16 份重复子树的访问数 2950→250，含准备时间仅从 13.243 降至 11.276 ms。 [题三回传、顺序与缓存对照](question3/README.md)

### 题四 · 宝藏分配

先计算 A、B、T1–T4 的点对距离，再按双方位置、剩余宝藏位掩码、累计收益、当前玩家展开博弈。每次取宝的净收益为宝藏价值减路费。H=2 时 T2 的即时净收益较有利，H=4 时先拿 T3 会改变对手后两步的去向与路费。原题 H=4 下，Minimax、标签顺序 Alpha-Beta、收益排序 Alpha-Beta 分别访问 65、47、37 个节点。

优化分开计量：寻路阶段比较逐点和多目标搜索，博弈阶段分别开关延迟创建后继与收益表。延迟构造在固定顺序下保留相同访问数，但省去被剪分支的对象创建，原题后继对象 44→36。另分别改变一个地形格费用与 T3 的价值，观察最短路改道、收益差和最优首步何时改变。 [题四矩阵、决策推演、消融与复跑命令](question4/README.md)

## 5. 结果与提交

### 报告图表与对应数据

在已按第 1 节安装依赖的环境中，从仓库根目录重建五幅图：

```powershell
& $plot report_figures.py --output .venv/repro/figures
```

该命令读取保存的数值并生成图片及 JSON。字体可能随环境变化，数值可对照原图附带的 JSON。

| 报告图 | 用来观察什么 | 数值 |
| --- | --- | --- |
| [图 1 · 迷宫搜索范围](report/figures_reproduced/q1_equal_f.png) | 对角空地上的同 `f` 候选为何会拖慢 FIFO | [数据](report/figures_reproduced/q1_data.json) |
| [图 2 · 加权路径与下界](report/figures_reproduced/q2_required_evidence.png) | 走 8 步为何只付 7，普通距离在哪里高估 | [数据](report/figures_reproduced/q2_data.json) |
| [图 3 · 样例剪枝位置](report/figures_reproduced/q3_pruning.png) | 左序下哪个叶子被跳过，为什么根值不变 | [数据](report/figures_reproduced/q3_data.json) |
| [图 4 · 不同深度的首步收益](report/figures_reproduced/q4_depth_choices.png) | T2、T3 的保证值何时改变排序 | [数据](report/figures_reproduced/q4_data.json) |
| [图 5 · 地形与宝藏价值](report/figures_reproduced/q4_controlled_compact.png) | 只改变一项时，路费何时进入平台、首步何时切换 | [数据](report/figures_reproduced/q4_controls_data.json) |

| 报告表 | 核对数据 |
| --- | --- |
| 表 1 · 迷宫扩展数 | [46 图配置与汇总](question1/results/course_followup/20261105T024931827094Z/summary.csv) |
| 表 2 · 加权地形正向扩展 | [9 种下界配置及建表成本](question2/results/geometry_followup/20261105T094155277386Z/summary.csv) |
| 表 3 · 博弈树访问顺序 | [节点、读叶与耗时](question3/results/report_followup/20261105T2245261782216/summary.csv) |
| 表 4、5 · 寻宝距离与逐回合收益 | [6×6 矩阵](question4/results/distance_matrix.csv) · [一条最优对局](question4/results/principal_variations.csv) |
| 表 6 · 深度、排序、缓存 | [访问数和终端评价数](question4/results/game_summary.csv) · [精确首步值与剪枝返回界限](question4/results/root_partial_returns.csv) |
| 表 7 · 等值启发的计算成本 | [逐目标扫描与启发值场](question4/results/heuristic_field/20261102T105518857084Z/summary.csv) |

### 交付文件

课程平台提交一个 ZIP 或 RAR，解压后直接得到《实验报告.pdf》。报告最多五页，首页链接到本仓库。四题的可核对输出分别是路径坐标、通行费、根得分，以及包含距离矩阵和最优策略的 JSON。官方样例的原始结果见 [运行输出记录](verification/20261109T025803635497Z/sample_runs.txt)。

## 6. 复现与限制

### 从输入重跑主要对照

在根目录运行。先设置 `$run`，再执行所需题目的命令。同名输出目录可能已经存在，重跑时重新设置 `$run`。各题 README 还列出其他对照、输出位置和计时范围。

```powershell
$run = Join-Path ".venv/repro" (Get-Date -Format "yyyyMMdd-HHmmss-fff")

python question1/heuristic_study.py --repeats 7 --output "$run/q1-heuristics"
python question2/geometry_heuristic_study.py --repeats 9 --output "$run/q2-heuristics"
python question2/batch_reverse_study.py --repeats 7 --output "$run/q2-batch"
python question2/batch_policy_study.py --repeats 7 --output "$run/q2-policy"
python question3/benchmark.py --repeats 8 --output "$run/q3-order"
python question3/audit_report_evidence.py --results "$run/q3-order" --study
python question3/structural_study.py --repeats 12 --output "$run/q3-cache"
python question4/benchmark.py --repeats 8 --output "$run/q4-game"
python question4/successor_cost_study.py --repeats 11 --batch 3 --output "$run/q4-construction"
python question4/heldout_successor_study.py --repeats 9 --batch 3 --output "$run/q4-new-maps"
python question4/heuristic_field_study.py --repeats 7
```

题四最后一个脚本在 `question4/results/heuristic_field/<时间戳>/` 创建新批次。如需从头计算图 5 的数据，在仓库副本中使用已安装绘图依赖的 Python 运行 `python question4/required_evidence.py --study`，再重画图。该命令会更新副本中对应的单变量结果文件。

### 核验指标与计时口径

- **先验算答案，再比较开销。** 检查路径是否经过合法四邻、累计费用是否正确，以及博弈算法是否返回相同根值。等长路径或等值首步可能有多个，核对代价或收益及最优集合。
- **区分计数。** A* 的扩展数只计有效出堆后实际检查邻居的状态，终点出堆即结束，计为一次取出而非扩展。博弈访问数含根，开启缓存时访问数含命中，另外统计终端评价、预读和后继构造。
- **比较同一批次。** `cases.json` 记录输入及适用的生成种子，`raw_runs.csv` 保存逐次测量，`summary.csv` 给出配置、计数和汇总，`environment.json` 记录运行环境、配置或源码指纹。题三和题四的基本随机顺序对照用种子 42，顺序敏感性另测多个种子。计时包含哪些准备步骤以该批记录为准。
- **检查复现差异。** 优先比对答案、扩展数、访问数和配置。耗时受系统、硬件和负载影响。需要追查差异时用 `--stats` 查看标准错误上的统计，用 `--trace-output` 保存轨迹，并确认样例和配置与表中一致。记录轨迹会增加开销，计时用独立脚本。
### 常见情况与处理

| 现象 | 如何核对 |
| --- | --- |
| Python 命令不存在或无法导入绘图库 | 用 `python --version` 确认解释器。绘图请用安装依赖时使用的同一个解释器运行 |
| JSON 解析失败、坐标越界或地形无效 | 对照 `sample.json` 的字段和类型，检查文件编码、地图矩形性及起终点。题二扩展可用零费用格，题四普通地形费用至少为 1 |
| 地图合法但目标不可达 | 题一返回 `[]`，题二返回 `-1`。题四的完整博弈要求关键点之间可达，遇不可达矩阵会报错 |
| 搜到的路径或并列首步与记录不同 | 对照同分顺序与参数，再核验路径是否合法、总代价是否最优，或该首步是否属于同一最优集合 |
| 运行开销与报告有差异 | 先比同一配置下的答案、扩展数和叶评价，再对照环境记录的准备成本与计时范围。报告耗时是所测场景在本机的观测值 |
| 深树或较多宝藏耗时明显增加 | 题三根到叶的实现上限为 400。题四可先用 `--matrix-only` 核对寻路，或用较小的 `--depth` 检查有限深度决策 |

核验范围为课程给出的样例和仓库自建测试。题二的另设地形、题三的树结构和题四的新地图各针对特定搜索行为，性能结论适用于这些输入及对应配置。
