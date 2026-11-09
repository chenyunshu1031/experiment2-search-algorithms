"""A 2x2 ablation: exact structural sharing versus bound-aware caching."""
import copy
from pathlib import Path
import sys

from experiment_cases import cases, balanced
from game_tree import TreeNode, graph_to_dict, inspect_graph, search
from reference import reference_value
from structural_cache import search_shared

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from course_study_io import paired_runs, save_study, study_args

CONFIGS = (('original',False,False,'identity'),('cache',False,True,'identity'),
           ('sharing',True,False,'identity'),('sharing_cache',True,True,'identity'),
           ('fused_keys',False,False,'structure'),('fused_cache',False,True,'structure'))


def extra_cases():
    motif = balanced(3,5,seed=711)
    for repeats in (2,4,8,16):
        # Independent objects in every occurrence, like an expanded JSON tree.
        yield dict(case=f'duplicate_motif_{repeats}',family='duplicate_motif',
                   root=TreeNode(children=[copy.deepcopy(motif) for _ in range(repeats)]))


def main():
    repeats,output = study_args(HERE)
    raw,manifest = [],[]
    inputs = list(cases())+list(extra_cases())
    for number,case in enumerate(inputs,1):
        root = case['root']
        expected = reference_value(root)
        info = inspect_graph(root)
        structural_objects = len(set(inspect_graph(root,structural_keys=True).structural_ids.values()))
        before = graph_to_dict(root)
        manifest.append(dict(case=case['case'],family=case['family'],expected=expected,graph=before))
        baseline = search(root)
        for repeat,(name,sharing,cache,cache_key) in paired_runs(CONFIGS,repeats+1):
            if sharing:
                result,preparation,elapsed = search_shared(root,use_cache=cache)
                canonical_objects = preparation.canonical_objects
                prep_ms = preparation.preparation_seconds*1000
            else:
                result = search(root,use_cache=cache,cache_key=cache_key)
                elapsed,prep_ms,canonical_objects = result.stats.elapsed_seconds,0,len(info.nodes)
                if cache_key=='structure':
                    canonical_objects = structural_objects
            assert result.value == expected,(case['case'],name)
            stats = result.stats
            if not cache:
                assert stats.visited_nodes == baseline.stats.visited_nodes
                assert stats.leaf_evaluations == baseline.stats.leaf_evaluations
            if repeat:
                raw.append(dict(case=case['case'],family=case['family'],mode=name,repeat=repeat,
                                value=expected,input_objects=len(info.nodes),canonical_objects=canonical_objects,
                                unfolded_nodes=stats.unfolded_nodes,visited=stats.visited_nodes,
                                search_leaves=stats.leaf_evaluations,pruned_leaves=stats.pruned_leaves,
                                cache_saved_leaves=stats.cache_saved_leaves,cache_hits=stats.cache_hits,
                                prep_ms=prep_ms,search_total_ms=stats.elapsed_seconds*1000,total_ms=elapsed*1000))
        assert graph_to_dict(root)==before
        if number%10==0 or number==len(inputs):
            print(f'{number}/{len(inputs)} inputs verified',flush=True)
    save_study(output,raw,manifest,[HERE/p for p in ('structural_cache.py','structural_study.py',
               'game_tree.py','reference.py','experiment_cases.py')]+[HERE.parent/'course_study_io.py'],
               'one-shot total: validation, optional separate sharing + canonical validation OR fused '
               'structural keys during validation, left-order Alpha-beta, bound-aware cache; oracle excluded. '
               'prep_ms records only separate sharing; fused key preparation is included in search_total_ms.',
               repeats=repeats,profiles=list(CONFIGS),input_count=len(inputs),
               warmup_per_configuration=1,order='left',cross_input_cache=False)


if __name__ == '__main__':
    main()
