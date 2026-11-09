# 题四 · 竞争环境下的宝藏分配

## 任务与运行

A 与 B 交替选取一个剩余宝藏，沿最低费用路径到达后领取。经过其他宝藏不会自动收集，玩家不阻挡通行，已领宝藏格仍可经过。取得宝藏的净收益为价值减路费，允许负收益。以 `U=Score_A−Score_B` 评价，A 最大化 U，B 最小化 U。 [课程样例及输入字段](sample.json) · [实现](treasure_game.py)

以下命令在本目录执行。前两条均输出矩阵及 H=2、H=4 的结果：

```powershell
python treasure_game.py --input sample.json --algorithm minimax
python treasure_game.py --input sample.json --algorithm alphabeta
python treasure_game.py --input sample.json --matrix-only
python treasure_game.py --input sample.json --depth 4 --order greedy --cache --lazy-children --stats
python -m unittest -v
```

## 路费与状态

每步按进入格收费，A、B 和宝藏格均为 1，因此曼哈顿距离是费用下界。点对 A* 默认求有向距离。若首尾格费用相同，沿同一路径反走只交换首尾收费，费用相等。样例的 `--pair-mode symmetric` 可把 30 次有向查询减为 15 次。下表是独立 Dijkstra 核验后的矩阵。 [矩阵 CSV](results/distance_matrix.csv)

| 出发 / 到达 | A | B | T1 | T2 | T3 | T4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0 | 12 | 10 | 4 | 8 | 8 |
| B | 12 | 0 | 8 | 8 | 4 | 10 |
| T1 | 10 | 8 | 0 | 6 | 6 | 12 |
| T2 | 4 | 8 | 6 | 0 | 6 | 6 |
| T3 | 8 | 4 | 6 | 6 | 0 | 6 |
| T4 | 8 | 10 | 12 | 6 | 6 | 0 |

状态保存双方位置、剩余宝藏位掩码、双方累计收益与当前玩家。A 首取 T2 时，位置变为 T2、清除其位、增加 `6−4=2` 的收益并交给 B。取一个宝藏算一层，到达指定深度或取完即停止。

搜索将已发生的收益差与未来收益增量分开。缓存键用双方位置、剩余宝藏、当前玩家与剩余深度，缓存值是未来增量或其界限。某步令收益差改变 `δ`，传给子节点的窗口是 `(α−δ, β−δ)`，回传时加回 `δ`。这样同一后续状态即使经过不同的历史收益也可复用。

## 深度与顺序对照

| 深度 | T1 首步 | T2 首步 | T3 首步 | T4 首步 | 最优选择 |
| --- | ---: | ---: | ---: | ---: | --- |
| H=2 | −4 | −1 | −2 | −4 | T2，−1 |
| H=4 | −5 | 1 | 3 | −7 | T3，3 |

表中是对手最优应对下 A 能保证的收益差，由排列穷举核验。H=2 时，A 拿 T2 得 2，B 拿 T3 得 3。H=4 时，A 虽然首取 T3 要亏 1，却抢走了 B 就近可得的宝藏。B 若取 T1，后续的一条最优对局是 A 再取 T2、B 最后取 T4。B 的最后一步付 12、亏 5，最终收益差为 3。 [首步精确值](results/root_action_values.csv) · [逐回合收益](results/principal_variations.csv) · [报告图 4](../report/figures_reproduced/q4_depth_choices.png)

固定距离矩阵，H=4 的 Minimax、标签顺序 Alpha-Beta、收益排序 Alpha-Beta 分别访问 65、47、37 个节点。收益排序再加缓存，仍访问 37 次，终端评价从 12 降到 8，因为缓存命中也算访问。固定收益排序、只改根的首个试探动作，T2 先试访问 37 次，T3 先试访问 31 次，T1 或 T4 先试均为 43 次。四种顺序都返回 T3、收益差 3。 [博弈基准汇总](results/game_summary.csv)

## 优化与单变量拓展

| 对照 | 结果与适用条件 |
| --- | --- |
| 延迟构造后继与预计算收益表 | 固定排序与输入，分别开关两项优化，并分开比较缓存开启和关闭。原题延迟构造让后继对象从 44 降到 36，访问量相同。无缓存时 37 组中 32 组更快，单用收益表仅 11 组更快。另用 8 张新图复测，关闭缓存的 16 组均更快，开启缓存时 15 组更快 |
| 多目标寻路与启发值场 | 逐目标扫描和两遍网格预计算使用相同的最近目标曼哈顿下界，逐例路径、扩展数相同。三张 16 宝藏地图含建表快约 26%–27%，稀疏开放图慢约 11%。节省的是估值计算量，收益取决于访问范围与目标数量 |
| 浅层搜索辅助深层排序 | 固定矩阵及缓存设置。原题 H=4 的收益排序和浅层辅助均访问 37 个深层节点，但后者另外访问 37 个浅层节点。将浅层成本计入后，本例不划算 |
| 单格通行费和 T3 价值 | 只改格 `(3,3)` 的费用 `w`，T2 到 T3 的距离为 `min(w+3,6)`。只改 T3 价值时距离不变，H=4 在价值 6 时两手并列，7 起选 T3，H=2 到 8 才改选 T3 |

[构造与收益表消融](results/successor_cost/20261106T021641697877Z) · [新图复测](results/heldout_successor/20261109T021529689579Z) · [启发值场批次](results/heuristic_field/20261102T105518857084Z) · [浅层排序批次](results/course_followup/20261104T065038961213Z) · [单变量 CSV](results/controlled_parameter_results.csv) · [报告图 5](../report/figures_reproduced/q4_controlled_compact.png)

## 复跑与数据

```powershell
$run = Join-Path "../.venv/repro" (Get-Date -Format "yyyyMMdd-HHmmss-fff")
python benchmark.py --repeats 8 --output "$run/q4-game"
python course_order_study.py --repeats 6 --output "$run/q4-shallow-order"
python successor_cost_study.py --repeats 11 --batch 3 --output "$run/q4-construction"
python heldout_successor_study.py --repeats 9 --batch 3 --output "$run/q4-new-maps"
python heuristic_field_study.py --repeats 7
```

博弈基准固定矩阵，分开看访问、叶评价和耗时。启发值场脚本读取保存的 [27 个输入场景](results/innovation/cases.json)，其中有 17 张不同地图，在 `results/heuristic_field/` 下生成时间戳目录。复跑后，可在仓库根目录用 `python check_clean_study.py <保存批次> <新批次>` 比对该实验的确定性计数。

如需重做多目标与逐点 A* 的原始对照，在安装绘图依赖的仓库副本运行 `python innovation_study.py --repeats 7`。它会更新该副本的输入与统计文件，原批次见[寻路与完整流程对照](results/innovation)。

只需重画报告图时使用根目录绘图入口。若要从输入重新计算单变量数据，在仓库副本安装绘图依赖后，于本目录运行 `python required_evidence.py --study`。它会更新副本中的单变量结果。寻宝题未定义不可达宝藏的博弈规则，正式入口遇到关键点不连通会报错。
