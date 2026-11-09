"""Compare safe geometry and terrain bounds with identical larger-g ordering."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

from course_axis_study import toll_cases
from reference import dijkstra_cost, distances_to_goal
from weighted_astar import DIRECTIONS, heuristic_parameters, normalize, prepare_heuristic, search
from weighted_heuristic_study import pocket_cases

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from course_study_io import paired_runs, save_study

CONFIGS = (
    ('zero', 'zero', 128),
    ('chebyshev', 'chebyshev', 128),
    ('euclidean', 'euclidean', 128),
    ('corrected', 'corrected', 128),
    ('goal-aware', 'goal-aware', 128),
    ('axis', 'axis', 128),
    ('reverse32', 'reverse', 32),
    ('reverse128', 'reverse', 128),
    ('reverse512', 'reverse', 512),
)


def audit(costs, start, end):
    minimum, near = heuristic_parameters(costs, end)
    exact = distances_to_goal(costs, end)
    functions = {name: prepare_heuristic(costs, end, minimum, near, mode, budget)[0]
                 for name, mode, budget in CONFIGS}
    point_checks = edge_checks = 0
    for r,row in enumerate(costs):
        for c,fee in enumerate(row):
            if fee is None:
                continue
            values = {name: h((r,c)) for name,h in functions.items()}
            assert values['chebyshev'] <= values['euclidean'] <= values['corrected'] <= values['goal-aware']
            for name,h in functions.items():
                value = values[name]
                assert 0 <= value <= exact[r][c], (name,(r,c),value,exact[r][c])
                point_checks += 1
                for dr,dc in DIRECTIONS:
                    nr,nc = r+dr,c+dc
                    if 0 <= nr < len(costs) and 0 <= nc < len(row) and costs[nr][nc] is not None:
                        assert value <= costs[nr][nc]+h((nr,nc)), (name,(r,c),(nr,nc))
                        edge_checks += 1
    assert all(h(end) == 0 for h in functions.values())
    return {name:h(start) for name,h in functions.items()}, point_checks, edge_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('at least 2 timed repeats required')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or HERE/'results'/'geometry_followup'/stamp
    base = HERE/'results/cases.json'
    candidates = json.loads(base.read_text(encoding='utf-8-sig')) + list(toll_cases()) + list(pocket_cases())
    cases, aliases, seen = [], {}, {}
    for case in candidates:
        key = json.dumps([case['grid'],case['start'],case['end']],sort_keys=True)
        if key in seen:
            aliases[case['case']] = seen[key]
        else:
            seen[key] = case['case']
            cases.append(case)
    raw, point_checks, edge_checks = [], 0, 0
    for number,case in enumerate(cases,1):
        inputs = case['grid'],case['start'],case['end']
        costs,start,end = normalize(*inputs)
        expected = dijkstra_cost(*inputs)
        case['dijkstra_cost'] = expected
        start_bounds,points,edges = audit(costs,start,end)
        point_checks += points
        edge_checks += edges
        for repeat,(name,mode,budget) in paired_runs(CONFIGS,args.repeats+1):
            result = search(*inputs,heuristic=mode,tie_break='larger-g',reverse_budget=budget)
            assert result.cost == expected, (case['case'],name,result.cost,expected)
            if expected < 0:
                assert not result.path
            else:
                assert result.path[0] == start and result.path[-1] == end
                assert all(0 <= r < len(costs) and 0 <= c < len(costs[r]) and costs[r][c] is not None
                           for r,c in result.path)
                assert all(abs(r-nr)+abs(c-nc) == 1 for (r,c),(nr,nc) in zip(result.path,result.path[1:]))
                assert sum(costs[r][c] for r,c in result.path[1:]) == expected
            if repeat:
                stats = result.stats
                raw.append(dict(case=case['case'],family=case['family'],mode=name,repeat=repeat,
                                cost=expected,start_h=start_bounds[name],expanded=stats.expanded,
                                reverse_settled=stats.reverse_settled,
                                total_nodes=stats.expanded+stats.reverse_settled,
                                heap_peak=stats.peak_frontier,total_ms=1000*stats.elapsed_seconds))
        if number % 10 == 0 or number == len(cases):
            print(f'verified and timed {number}/{len(cases)} unique inputs',flush=True)
    sources = [HERE/p for p in ('geometry_heuristic_study.py','weighted_astar.py','axis_heuristic.py',
                                'reverse_bound.py','reference.py','weighted_heuristic_study.py','course_axis_study.py')]
    sources.append(HERE.parent/'course_study_io.py')
    save_study(output,raw,cases,sources,
               'normalization + heuristic preparation + A* + reconstruction; independent checks excluded',
               repeats=args.repeats,tie_break='larger-g',warmup_per_configuration=1,
               candidate_scenarios=len(candidates),unique_inputs=len(cases),duplicate_aliases=aliases,
               point_checks=point_checks,edge_checks=edge_checks,
               base_cases_sha256=sha256(base.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
