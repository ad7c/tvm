"""
Experiment 02c: Compare pass rates — full space vs single pruning vs double pruning.
Produces chart: figures/exp02_comparison.png

Reads: data/exp02_verify.json (if exists) or uses model-only estimates.
"""
import sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, 'F:/tvm/experiments')
from common import enumerate_pruned_space, get_divisors


def load_or_compute():
    """Load verify data if available, or compute from model."""
    verify_path = 'F:/tvm/experiments/data/exp02_verify.json'
    model_path = 'F:/tvm/experiments/data/exp02_model.json'

    verify = None
    model = None

    if os.path.exists(verify_path):
        with open(verify_path) as f:
            verify = json.load(f)
    if os.path.exists(model_path):
        with open(model_path) as f:
            model = json.load(f)

    return verify, model


def create_chart(verify=None, model=None):
    od = 'F:/tvm/experiments/figures'
    os.makedirs(od, exist_ok=True)

    # ── Model data ──
    if model and '256x256x256' in model:
        m = model['256x256x256']
        full_n = m['full']
        single_n = m['single']
        double_n = m['double']
    else:
        full_n = len(get_divisors(256)) ** 3
        single_n = len(enumerate_pruned_space(256, 256, 256, mode='single'))
        double_n = len(enumerate_pruned_space(256, 256, 256, mode='double'))

    # ── Verify data ──
    verify_pass = None
    if verify:
        verify_pass = {}
        for label in ['outside', 'single', 'double']:
            if label in verify:
                pool = verify[label]
                n = len(pool)
                n_pass = sum(1 for r in pool if r.get('tvm_pass'))
                verify_pass[label] = {'n': n, 'pass': n_pass, 'rate': n_pass/n*100 if n else 0}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax1, ax2 = axes

    # ── Left: Space size comparison ──
    x = np.arange(3)
    groups = ['Full Space\n(no pruning)', 'Single Pruning\n(upward only)', 'Double Pruning\n(up+down)']
    totals = [full_n, single_n, double_n]
    colors_model = ['#95a5a6', '#f39c12', '#27ae60']

    bars = ax1.bar(x, totals, color=colors_model, width=0.5, edgecolor='white')
    for i, (bar, v, total) in enumerate(zip(bars, totals, [full_n]*3)):
        rate = v / full_n * 100
        ax1.text(i, v + full_n * 0.02, f'{v:,}\n({rate:.2f}%)',
                ha='center', fontsize=11, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=10)
    ax1.set_title(f'Search Space Size Reduction\nMatMul 256x256x256 on RTX 3050',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Block-Tile Configurations', fontsize=11)

    # ── Right: Pass rate comparison (if verify data available) ──
    if verify_pass:
        labels = ['Outside\nPruning', 'Single\nPruning', 'Double\nPruning']
        rates = [verify_pass['outside']['rate'],
                 verify_pass['single']['rate'],
                 verify_pass['double']['rate']]
        sizes = [verify_pass['outside']['n'],
                 verify_pass['single']['n'],
                 verify_pass['double']['n']]
        passes = [verify_pass['outside']['pass'],
                  verify_pass['single']['pass'],
                  verify_pass['double']['pass']]
        colors_v = ['#e74c3c', '#f39c12', '#27ae60']

        ax2.bar(range(3), rates, color=colors_v, width=0.5)
        for i, (r, s, p) in enumerate(zip(rates, sizes, passes)):
            ax2.text(i, r + 2, f'{p}/{s}\n({r:.1f}%)',
                    ha='center', fontsize=11, fontweight='bold')
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_title('TVM Postproc Pass Rate by Group', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Pass Rate (%)', fontsize=11)
        ax2.set_ylim(0, max(rates) * 1.5 if max(rates) > 0 else 100)
    else:
        ax2.text(0.5, 0.5, 'Run exp02/verify.py first\nto get pass rate data',
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_title('Pass Rate (pending verification)', fontsize=13)

    plt.tight_layout()
    plt.savefig(f'{od}/exp02_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {od}/exp02_comparison.png')


if __name__ == '__main__':
    verify, model = load_or_compute()
    create_chart(verify, model)
