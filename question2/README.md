# 题二 加权地形

用 A* 求最小通行费。每一步支付进入格的费用，起点初始不收费，进入终点 E 也免费。输入 JSON 含 `grid`、`start`、`end`，坐标从 0 开始。输出整数费用，样例为 `7`，无解为 `-1`。 [样例输入](sample.json)

以下命令在本目录运行。

```powershell
python weighted_astar.py --input sample.json
python weighted_astar.py --input sample.json --heuristic zero --stats --details
python weighted_astar.py --input sample.json --heuristic goal-aware --stats --details
python -m unittest -v
```

样例走 8 步，前 7 步各付 1，最后一步付 0。默认启发函数扣除免费一步，并在适用时计入进入终点邻格的最低费用。固定较大 `g` 优先，零启发扩展 14 格，修正下界扩展 8 格。 [逐格费用](results/official_path_costs.csv)

```powershell
python geometry_heuristic_study.py --repeats 9 --output ../.venv/repro/q2-heuristics
python batch_reverse_study.py --repeats 7 --output ../.venv/repro/q2-batch
```

几何下界对照固定同分顺序，独立 Dijkstra 核验代价。批量对照固定地图和终点，比较逐起点 A*、共享反向下界的 A* 和一次精确反向 Dijkstra，计时含准备和路径恢复。反向沿原边的收费方向计算，并为不可达起点返回 `-1` 与空路径。单次短查询与多次查询的耗时分别比较。 [几何对照](results/geometry_followup/20261105T094155277386Z/summary.csv) · [批量对照](results/course_followup/20261109T031416794508Z/summary.csv)
