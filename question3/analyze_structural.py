"""Summarize structural-cache ablations from saved CSV, without rerunning."""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ('original','cache','sharing','sharing_cache','fused_keys','fused_cache')
LABELS = ('原始左序','对象缓存','单独合并无缓存','单独合并＋缓存','验证时生成结构标识','结构标识缓存')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args = parser.parse_args()
    folder = args.results
    records = list(csv.DictReader((folder/'summary.csv').open(encoding='utf-8-sig')))
    metadata = json.loads((folder/'environment.json').read_text(encoding='utf-8'))
    data = {(r['case'],r['mode']):r for r in records}
    cases = sorted({r['case'] for r in records})
    selected = ('official_json','equal_b3_d6','increasing_b3_d6','dag_w8_d9_0',
                'duplicate_motif_8','duplicate_motif_16')
    lines = ['# 题三结构标识缓存的对照与要求核对','',
             '研究问题：普通JSON中的相同子树是不同对象，按对象身份缓存不能复用。能否识别结构相同的子树，同时保留左右顺序和最佳得分？','',
             '## 算法与正确性','',
             '叶子用“数值类型、数值”作标识，内部节点用按原顺序排列的子标识作标识。字典按完整键比较，不把相同根得分或单独的哈希值当成相同结构。',
             '由叶向上归纳，标识相同的子树在相同MAX/MIN角色下有相同Minimax值。缓存键还包含当前玩家，并保存EXACT/LOWER/UPPER；上下界只收紧窗口，窗口未关闭则继续搜索。',
             '识别结构不计算Minimax真值，不重排children，也不修改输入。单独复制合并与在原有验证中计算标识，两版识别同样的结构，搜索访问数和叶评估一致。',
             '期望字典操作下，结构识别为O(V+E)时间、O(V+E)额外键存储；显式输入仍须全部验证，搜索叶评估减少不等于完全不读取输入叶分。','',
             '## 实验设置','',
             '沿用38个固定输入，另用同一个随机子树构造2/4/8/16份独立副本，构成42输入。副本数是单变量，副本不共享Python对象。',
             f"六配置统一从左到右，分别预热一次并轮换计时{metadata['repeats']}次，共{metadata['timed_runs']}次。四配置拆分“单独结构合并×缓存”，两配置检验“把结构标识计算并入验证×缓存”。所有准备、验证、搜索与缓存开销都计入总耗时；缓存每次重置。",
             '根值由独立完整回传核验，关闭缓存时要求原树、合并树、仅生成标识三者的访问数与叶评估相同。测试还核对逐次返回值的上下界、不同玩家角色、输入不变和左右顺序。','',
             '## 代表结果','',
             '下表为访问数／搜索叶评估／总耗时中位数ms。时间只使用本批数据，不与之前的排序批次混合。','',
             '|输入|'+'|'.join(LABELS)+'|','|---|'+'---|'*len(MODES)]
    for case in selected:
        lines.append('|'+case+'|'+'|'.join(f"{data[case,m]['visited']} / {data[case,m]['search_leaves']} / {float(data[case,m]['total_ms']):.3f}" for m in MODES)+'|')
    faster = sum(float(data[n,'fused_cache']['total_ms'])<float(data[n,'original']['total_ms']) for n in cases)
    def ms(case,mode):
        return f"{float(data[case,mode]['total_ms']):.3f}"
    lines += ['',
              f"16份副本中，结构标识缓存使访问2950→250、叶评估1733→112，含识别总耗时{ms('duplicate_motif_16','original')}→{ms('duplicate_motif_16','fused_cache')}毫秒；单独复制合并后再缓存为{ms('duplicate_motif_16','sharing_cache')}毫秒。两种结构复用方法的访问数相同，融合识别省去单独复制和再次验证。",
              f"无重复的递增树，搜索访问仍为325，耗时为{ms('increasing_b3_d6','original')}→{ms('increasing_b3_d6','fused_cache')}毫秒。小样例叶评估7→6，总耗时{ms('official_json','original')}→{ms('official_json','fused_cache')}毫秒；全同分树搜索125→19次，总耗时{ms('equal_b3_d6','original')}→{ms('equal_b3_d6','fused_cache')}毫秒。应结合总时间判断输入扫描和标识计算是否抵消了搜索节省。",
              f"本批结构标识缓存在{faster}/{len(cases)}个输入上中位耗时小于原始左序，不能作为通用最快方案。已有对象共享的DAG用对象缓存{ms('dag_w8_d9_0','cache')}毫秒，结构缓存{ms('dag_w8_d9_0','fused_cache')}毫秒，亦须比较额外识别的成本。",
              '保留原始左序为正式默认，结构标识缓存仅用于重复结构较多的扩展。小输入的时间区间有重叠，不把微小差异说成稳定优势；重复副本场景是机制验证，不代表真实游戏状态分布。',
              '', '## 与题目三要求逐项对应','',
              '|要求|证据|判断|','|---|---|---|',
              '|根MAX、逐层递归、从左到右、返回最佳得分|game_tree.py的alpha_beta；样例根得分5|已完成，默认不启用排序或缓存|',
              '|Minimax数值回传|底层5、9、9、2，中层5、2，根5；official_process.json|已完成|',
              '|剪枝依据与状态|局部窗口(−∞,5)，读6后α≥β，跳过9；返回6只是下界|已完成|',
              '|比较不同子节点顺序开销|左/逆/随机/探测、搜索与探测读取、总耗时；24种随机顺序|已完成，排序仅作拓展|',
              f"|自设对照与算法提升|结构合并×缓存的拆分、融合标识、{len(cases)}输入与{metadata['timed_runs']}次计时|已补充|",
              '|结果分析与独立思考|无重复、小样例、同分树中均可能变慢；保留准备成本|已补充适用边界|',
              '|图表支撑|Word图3和表3，本文六配置表，CSV与输入|已完成|',
              '|A*的g/h/f及优先队列|属于题一二；题三使用递归窗口|不硬套到题三|',
              '', '## 复现','',
              '`python structural_study.py --repeats 12 --output <新的结果目录>`',
              '`python analyze_structural.py --results <该目录>`',
              '`python game_tree.py --cache --cache-key structure --stats`',
              '所有原始计时、输入、环境、源码指纹保存在本目录；raw_runs.csv的prep_ms仅列单独合并成本，融合识别已包含在search_total_ms中。',
              'canonical_objects在融合模式中表示结构标识种类数，输入对象仍全部保留，不应解读成复制后对象数或实际内存字节。']
    (folder/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(folder/'analysis.md')


if __name__=='__main__':
    main()
