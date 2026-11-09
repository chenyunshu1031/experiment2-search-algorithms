# 题四 宝藏分配

先用 A* 计算 A、B 和四个宝藏之间的最小路费，再用 Minimax 和 Alpha-Beta 计算 A 的最优首步。输入 JSON 含 `grid`、`a_start`、`b_start`、`treasures`，每个宝藏有名称、位置和价值。每领取一个宝藏算一层，净收益为价值减路费，双方轮流行动。 [样例输入](sample.json)

以下命令在本目录运行。前两条各输出一个 6×6 距离矩阵和两个深度的结果。H=2 选 T2，收益差 −1。H=4 选 T3，收益差 3。

```powershell
python treasure_game.py --input sample.json --algorithm minimax
python treasure_game.py --input sample.json --algorithm alphabeta
python treasure_game.py --input sample.json --matrix-only
python treasure_game.py --depth 4 --order greedy --cache --lazy-children --stats
python -m unittest -v
```

状态保存双方位置、剩余宝藏、累计收益与当前玩家。距离矩阵由独立 Dijkstra 核验，博弈结果由排列穷举核验。固定收益排序比较提前构造与延迟构造，可单独看减少后继对象分配的收益。 [距离矩阵](results/distance_matrix.csv) · [首步价值](results/root_action_values.csv) · [一条最优对局](results/principal_variations.csv)

```powershell
python benchmark.py --repeats 8 --output ../.venv/repro/q4-game
python successor_cost_study.py --repeats 11 --batch 3 --output ../.venv/repro/q4-construction
python heldout_successor_study.py --repeats 9 --batch 3 --output ../.venv/repro/q4-new-maps
python heuristic_field_study.py --repeats 7
```

启发值场脚本在 `results/heuristic_field/` 下创建新批次。单变量实验固定其余条件，分别修改一格的通行费或一个宝藏的价值。若要从头计算，可在仓库副本安装根目录的绘图依赖后运行 `python required_evidence.py --study`。 [单变量数据](results/controlled_parameter_results.csv) · [启发值场对照](results/heuristic_field/20261102T105518857084Z/summary.csv) · [延迟构造复测](results/heldout_successor/20261109T021529689579Z/summary.csv)
