# 题一 迷宫寻路

在四邻网格上用 A* 求最少步数。每走一步费用为 1，`X` 为障碍。输入 JSON 包含 `grid`、`start`、`end`，坐标按从 0 开始的行、列表示。输出包含起点和终点的路径坐标，无路时输出 `[]`。 [样例输入](sample.json)

以下命令在本目录运行。若从仓库根目录运行，给脚本和输入路径加上 `question1/`。

```powershell
python maze_astar.py --input sample.json
python maze_astar.py --input sample.json --heuristic zero --tie-break fifo --stats
python maze_astar.py --input sample.json --heuristic manhattan --tie-break fifo --stats
python -m unittest -v
```

样例从 `(0,0)` 到 `(3,4)`，走 7 步。固定 FIFO 时，零启发扩展 16 格，曼哈顿扩展 13 格，最短路均为 7 步。扩展数不含取出后直接结束的终点。可用 `--trace-output trace.json` 查看每一步的出堆和更新。 [样例搜索轨迹](results/traces/official_manhattan_fifo.json)

```powershell
python heuristic_study.py --repeats 7 --output ../.venv/repro/q1
```

实验包含官方样例、对角空地、走廊、隔墙、无解与随机地图。固定 FIFO 比较启发函数，再固定曼哈顿比较同分顺序。独立 BFS 核验最短步数。隔墙下界对显著绕行有帮助，简单地图上的预处理可能增加耗时。 [保存的计时汇总](results/course_followup/20261105T024931827094Z/summary.csv)
