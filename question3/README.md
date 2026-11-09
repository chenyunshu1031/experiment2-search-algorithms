# 题三 博弈树

输入 JSON 的 `tree_structure` 是一棵树，内部节点用 `children` 保存子节点，叶子用 `value` 保存收益。根为 MAX，下一层为 MIN。两个算法均输出根值，样例为 `5`。 [样例输入](sample.json)

以下命令在本目录运行。题设左序对应默认的 `--order left`。

```powershell
python game_tree.py --input sample.json --algorithm minimax --stats
python game_tree.py --input sample.json --algorithm alphabeta --stats
python game_tree.py --input sample.json --order reverse --stats
python game_tree.py --input sample.json --order random --seed 42 --stats
python game_tree.py --input sample.json --order probe --stats
python -m unittest -v
```

样例中，Minimax 访问 15 个节点并读取 8 个叶子，左序 Alpha-Beta 为 14 个节点和 7 个叶子。首叶探测的开销包含搜索读叶与排序预读。可用 `--trace-output trace.json` 保存实际窗口与剪枝事件。 [左序轨迹](results/traces/official_json_ab_left.json)

```powershell
python benchmark.py --repeats 8 --output ../.venv/repro/q3-order
python audit_report_evidence.py --results ../.venv/repro/q3-order --study
python structural_study.py --repeats 12 --output ../.venv/repro/q3-cache
```

结构缓存用有序子树内容识别重复分支，并区分精确值、下界和上界。无重复子树时，结构识别也要花时间。 [顺序对照](results/report_followup/20261105T2245261782216/summary.csv) · [结构缓存对照](results/structural_followup/20261105T2239574201109/summary.csv)
