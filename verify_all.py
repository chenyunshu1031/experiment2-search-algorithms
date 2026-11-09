"""Run four isolated test suites and assignment examples; no benchmarks."""
import argparse
import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY = ('maze_astar.py', 'weighted_astar.py', 'game_tree.py', 'treasure_game.py')


def run(folder, args):
    result = subprocess.run([sys.executable, *args], cwd=folder, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=180)
    if result.returncode:
        raise RuntimeError(f'{folder.name} {args}\n{result.stdout}\n{result.stderr}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    rows = []
    for number, entry in enumerate(ENTRY, 1):
        folder = ROOT / f'question{number}'
        result = run(folder, [entry])
        answer = json.loads(result.stdout) if number == 4 else ast.literal_eval(result.stdout)
        if number == 1:
            assert len(answer) == 8 and tuple(answer[0]) == (0,0) and tuple(answer[-1]) == (3,4)
        elif number == 2: assert answer == 7
        elif number == 3: assert answer == 5
        else:
            assert len(answer['distance_matrix']) == 6
            assert all(len(row)==6 for row in answer['distance_matrix'])
            assert answer['H=2'] == dict(best_target='T2',best_value=-1)
            assert answer['H=4'] == dict(best_target='T3',best_value=3)
        tests = run(folder, ['-m', 'unittest', '-v']) if not args.smoke_only else None
        count = int(re.search(r'Ran (\d+) tests?', tests.stderr)[1]) if tests else 0
        rows.append(dict(question=number, official_output=answer, tests=count,
                         test_log=tests.stderr if tests else 'not run'))
        print(f'question{number}: official sample passed; {count} tests passed', flush=True)
    output = ROOT / 'verification' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    sources = sorted(p for p in ROOT.rglob('*.py') if 'verification' not in p.parts)
    report = dict(python=platform.python_version(), platform=platform.platform(),
                  smoke_only=args.smoke_only, tests=sum(r['tests'] for r in rows), questions=rows,
                  source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  scope='Self-built tests and provided examples only; no claim about hidden grader tests.')
    (output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'All passed. Evidence: {output}')


if __name__ == '__main__': main()
