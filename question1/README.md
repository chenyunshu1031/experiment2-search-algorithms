# 题一 · 迷宫寻路

## 任务与运行

在四邻网格上求最少步数。输入 JSON 包含 `grid`、`start`、`end`，坐标按从 0 开始的行、列表示。每步代价 1，`X` 为障碍。输出包含起点和终点的坐标序列，若无路则输出 `[]`，起终点相同则返回只含起点的路径。 [课程样例](sample.json) · [实现](maze_astar.py)

以下命令在本目录执行：

```powershell
python maze_astar.py --input sample.json
python maze_astar.py --input sample.json --heuristic zero --tie-break fifo --stats
python maze_astar.py --input sample.json --heuristic manhattan --tie-break fifo --stats --trace-output trace.json
python -m unittest -v
```

## 算法与样例

`g` 记录从起点走到当前格的最少已知步数，`h` 估计到终点的剩余步数，最小堆先取 `f=g+h` 较小的格。发现更小的 `g` 才更新前驱和堆，已经过期的堆项跳过。终点有效出堆后沿前驱回溯。曼哈顿距离等于行差与列差之和，每走一步最多减少 1，因此是可采纳且一致的下界。同分时优先较大的 `g` 只改变 `f` 相同的候选次序。

样例起点 `(0,0)`、终点 `(3,4)`，至少要走 7 步。FIFO 下，起点与首层两个候选的 `f` 都为 7。搜索先试向下的支路，但 `(3,3)` 为墙。走到 `(2,4)` 后，以 `g=7` 生成终点。第 14 次有效出堆时取到终点，回溯得到：

```text
(0,0) → (0,1) → (0,2) → (1,2) → (1,3) → (1,4) → (2,4) → (3,4)
```

路径达到 7 步下界。终点取出即结束，所以该轨迹的扩展数为 13。 [FIFO 出堆与更新轨迹](results/traces/official_manhattan_fifo.json)

## 对照与解释

46 张图包含官方样例、空地、走廊、隔墙、无解及随机图。先固定 FIFO 比较零启发、曼哈顿、向上取整欧氏及切比雪夫距离，再固定曼哈顿比较同分顺序。每个配置测 7 次，最短步数由 BFS 独立核验。

| 场景 | 零启发 + FIFO | 曼哈顿 + FIFO | 曼哈顿 + 较大 g |
| --- | ---: | ---: | ---: |
| 官方样例 | 16 | 13 | 13 |
| 31×31 对角空地 | 960 | 960 | 60 |
| 31×31 同行空地 | 720 | 30 | 30 |
| 61×61 随机图 | 3000 | 1822 | 264 |

表中单位为扩展格。样例用曼哈顿时，`(4,2)`、`(0,4)` 的 `f` 从 6 升到 9，排到终点的 `f=7` 之后，对应支路无需扩展。对角空地恰好相反，沿最短前缀到达的格子都可有 `f=60`，FIFO 会把同分候选逐层处理。改为较大 `g` 后优先向终点推进，耗时中位数 1.216→0.138 ms，堆峰值却从 31 增到 59，说明扩展量与内存峰值要分别看。 [报告图 1](../report/figures_reproduced/q1_equal_f.png)

隔墙下界保留一条带缺口的墙，去掉其他障碍，计算松弛图中的距离。101×101 绕墙图里，起点曼哈顿为 100 步，绕缺口至少要 300 步。固定较大 `g`，用隔墙下界后扩展 5151→300，含建表耗时 8.823→1.278 ms。46 张图中只有 6 张少扩展，41 张耗时中位数上升。显著的隔墙绕行更可能抵消准备成本。

## 复跑与数据

```powershell
$run = Join-Path "../.venv/repro" (Get-Date -Format "yyyyMMdd-HHmmss-fff")
python heuristic_study.py --repeats 7 --output "$run/q1-heuristics"
```

比较同一场景、同一同分规则下的 `expanded` 和计时中位数。随机地图种子见固定输入。 [输入与种子](results/course_followup/20261105T024931827094Z/cases.json) · [逐次计时](results/course_followup/20261105T024931827094Z/raw_runs.csv) · [汇总](results/course_followup/20261105T024931827094Z/summary.csv) · [环境及计时范围](results/course_followup/20261105T024931827094Z/environment.json)
