"""Save paired benchmark evidence in a separate timestamped directory."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from statistics import median


def study_args(folder):
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=6)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('至少重复 2 次。')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    return args.repeats, args.output or folder / 'results' / 'course_followup' / stamp


def paired_runs(configs, repeats):
    for repeat in range(repeats):
        offset = repeat % len(configs)
        for config in configs[offset:] + configs[:offset]:
            yield repeat, config


def save_study(output, raw, cases, sources, timing, **extra):
    groups = {}
    for row in raw:
        key = row['case'], row.get('depth'), row.get('cache'), row['mode']
        groups.setdefault(key, []).append(row)
    summary = []
    for rows in groups.values():
        fixed = {k: v for k, v in rows[0].items() if k != 'repeat' and not k.endswith('_ms')}
        assert all(all(r[k] == v for k, v in fixed.items()) for r in rows)
        times = [r['total_ms'] for r in rows]
        medians = {k: median(r[k] for r in rows) for k in rows[0] if k.endswith('_ms')}
        summary.append(dict(**fixed, **medians, min_ms=min(times), max_ms=max(times), repeats=len(rows)))
    output.mkdir(parents=True, exist_ok=False)
    for name, records in [('raw_runs', raw), ('summary', summary)]:
        with (output / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    case_data = json.dumps(cases, ensure_ascii=False, indent=2).encode('utf8')
    (output / 'cases.json').write_bytes(case_data)
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    processor=os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),
                    cases_sha256=sha256(case_data).hexdigest(), timed_runs=len(raw), timing_scope=timing,
                    sources={p.name: sha256(p.read_bytes()).hexdigest() for p in sources}, **extra)
    (output / 'environment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'{len(groups)} 组对照，{len(raw)} 次计时通过核验。\n{output.resolve()}')
