"""
Experiment 02a: Analytical pruning model.
Enumerate divisor space, apply single/double pruning, report space sizes.
Output: data/exp02_model.json
"""
import sys, os, json
sys.path.insert(0, 'F:/tvm/experiments')
from common import (
    get_divisors, enumerate_pruned_space,
    check_single_pruning, check_double_pruning,
    GPU,
)


def run(workloads=None):
    if workloads is None:
        workloads = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]

    results = {}
    for m, n, k in workloads:
        full = len(get_divisors(m)) * len(get_divisors(n)) * len(get_divisors(k))
        single_space = enumerate_pruned_space(m, n, k, mode='single')
        double_space = enumerate_pruned_space(m, n, k, mode='double')

        key = f'{m}x{n}x{k}'
        results[key] = {
            'full': full,
            'single': len(single_space),
            'double': len(double_space),
            'single_rate': len(single_space) / full * 100,
            'double_rate': len(double_space) / full * 100,
            'single_candidates': single_space[:20],
            'double_candidates': double_space[:20],
        }

    # Print table
    print(f'{"Workload":12s} {"Full":>8s} {"Single":>8s} {"Double":>8s} {"S%":>7s} {"D%":>7s}')
    print('-' * 54)
    for key, r in results.items():
        print(f'{key:12s} {r["full"]:>8,} {r["single"]:>8,} {r["double"]:>8,} '
              f'{r["single_rate"]:>6.1f}% {r["double_rate"]:>6.1f}%')

    return results


if __name__ == '__main__':
    results = run()
    od = 'F:/tvm/experiments/data'
    os.makedirs(od, exist_ok=True)
    with open(f'{od}/exp02_model.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {od}/exp02_model.json')
