# 题三 · Alpha-Beta 剪枝

## 任务与运行

输入 JSON 的 `tree_structure` 是博弈树。内部节点用 `children` 表示子节点，叶子用 `value` 表示得分。根为 MAX，下一层为 MIN，逐层交替。正式接口默认按 `children` 从左到右，输出一个根值。 [课程样例](sample.json) · [实现](game_tree.py)

以下命令在本目录执行：

```powershell
python game_tree.py --input sample.json --algorithm minimax --stats
python game_tree.py --input sample.json --algorithm alphabeta --order left --stats --trace-output trace.json
python game_tree.py --input sample.json --order reverse --stats
python game_tree.py --input sample.json --order random --seed 42 --stats
python game_tree.py --input sample.json --order probe --stats
python game_tree.py --input sample.json --cache --cache-key structure --stats
python -m unittest -v
```

## 样例回传与剪枝

完整样例有 15 个节点、8 个叶位置。四组叶子 `(3,5)`、`(6,9)`、`(6,9)`、`(1,2)` 分别回传 5、9、9、2。两个 MIN 分别取 5、2，根 MAX 取 5。

Alpha-Beta 根窗口从负无穷到正无穷。左 MIN 先得到 5，将 `β` 降到 5。下一棵 MAX 子树读到 6 后已有 `α≥β`，这棵子树继续搜索也只可能给出 6 或更高，左 MIN 已有更低的 5，因而跳过其余叶子。这里返回的 6 是下界，真实子树值为 9。若把下界当精确值写入缓存，之后换窗口就可能算错，自设反例中根真值 8 会被算成 6。 [左序轨迹](results/traces/official_json_ab_left.json) · [报告图 3](../report/figures_reproduced/q3_pruning.png)

## 访问顺序与缓存

表中为同一官方 JSON、关闭缓存的配对结果。时间为该批中位数，含验证与排序。

| 方法 | 访问数 | 搜索读叶 | 排序预读 | 总读叶 | ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Minimax 左序 | 15 | 8 | 0 | 8 | 0.043 |
| Alpha-Beta 左序 | 14 | 7 | 0 | 7 | 0.048 |
| Alpha-Beta 逆序 | 14 | 7 | 0 | 7 | 0.047 |
| Alpha-Beta 随机，种子 42 | 11 | 5 | 0 | 5 | 0.084 |
| Alpha-Beta 首叶探测 | 13 | 6 | 14 | 20 | 0.055 |

仅交换右 MIN 的两个孩子，若先读到 2，就有 `β=2≤α=5`，整棵 `(6,9)` 子树可跳过。样例的节点数 14→11，读取叶子 7→5。全树逆序仍读 7 叶，说明节省来自能否先遇到收紧窗口的子树。另在 4 个输入上测 24 种随机顺序，样例读叶在 5–8 次之间。

结构缓存按叶值和有序子结构识别重复子树，并区分精确值、下界和上界。普通 JSON 中，长得相同的子树是不同对象，按对象身份无法合并它们的结果。16 份重复子树场景的访问数 2950→250，含验证与标识准备的耗时 13.243→11.276 ms。无重复树时耗时 2.076→2.696 ms，结构识别的成本没有得到复用回报。

## 复跑与数据

```powershell
$run = Join-Path "../.venv/repro" (Get-Date -Format "yyyyMMdd-HHmmss-fff")
python benchmark.py --repeats 8 --output "$run/q3-order"
python audit_report_evidence.py --results "$run/q3-order" --study
python structural_study.py --repeats 12 --output "$run/q3-cache"
```

基准为 38 个输入、8 配置，每个计时 8 次，固定随机顺序种子 42，与独立完整回传核对。审计另外输出窗口轨迹和 24 种顺序的汇总。结构实验为 42 个输入、6 配置，每个计时 12 次，含重复与无重复输入。 [基准输入及计时](results/report_followup/20261105T2245261782216) · [随机顺序汇总](results/report_followup/20261105T2245261782216/order_summary.csv) · [结构缓存批次](results/structural_followup/20261105T2239574201109)
