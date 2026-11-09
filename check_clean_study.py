"""Compare saved deterministic results from two study directories."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original',type=Path)
    parser.add_argument('reproduced',type=Path)
    args=parser.parse_args()
    def read(folder):
        with (folder/'summary.csv').open(encoding='utf-8-sig') as stream:
            return {(r['case'],r['config']):r for r in csv.DictReader(stream)}
    a,b=read(args.original),read(args.reproduced)
    assert a.keys()==b.keys()
    fields=('points','expanded','generated','search_calls','heuristic_distance_terms','heuristic_field_cells')
    assert all(a[key][f]==b[key][f] for key in a for f in fields)
    report=dict(original=str(args.original),reproduced=str(args.reproduced),
                configurations=len(a),fields=list(fields),deterministic_match=True,
                timing_comparison='Not required to match; timing varies with load and environment.')
    out=Path(__file__).resolve().parent/'verification/clean_field_reproduction.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(a)} case/config pairs reproduced with matching deterministic counts:',out)


if __name__=='__main__':main()
