"""
Experiment 02b: Verify analytical pruning by testing specific tile parameters
against actual TVM postproc via trace JSON modification.

Output: data/exp02_verify.json
"""
import sys, os, time, json
from collections import Counter

sys.path.insert(0, 'F:/tvm/experiments')
from common import (
    init_tvm, get_postprocs, apply_postprocs,
    make_matmul_module, make_target,
    enumerate_pruned_space, check_single_pruning, check_double_pruning,
)

init_tvm()

from tvm.s_tir.schedule import Schedule
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply
from tvm.s_tir.schedule._ffi_api import TraceApplyJSONToSchedule


def run(n_per_group=10):
    """Test tile configs from each pruning group against TVM postproc."""
    mod = make_matmul_module()
    target = make_target()
    pps, pp_names = get_postprocs()
    print(f'Postproc: {" -> ".join(pp_names)}')

    # Get base trace for JSON modification
    sg = PostOrderApply()
    ctx = TuneContext(mod=mod, target=target, space_generator=sg, task_name='verify')
    ctx.space_generator._initialize_with_tune_context(ctx)
    ds_list = ctx.space_generator.generate_design_space(mod)
    base_json = ds_list[0].trace.as_json()
    instructions = base_json[0]
    decisions = base_json[1]

    # Find SP/RD instruction indices
    sp_idx = [i for i, inst in enumerate(instructions) if inst[0] == 'SamplePerfectTile' and inst[2][0] == 5]
    rd_idx = [i for i, inst in enumerate(instructions) if inst[0] == 'SamplePerfectTile' and inst[2][0] == 3]
    print(f'SP indices: {sp_idx}, RD indices: {rd_idx}')
    assert len(sp_idx) >= 2 and len(rd_idx) >= 1

    # Get candidate pools
    single_pool = enumerate_pruned_space(256, 256, 256, mode='single')
    double_pool = enumerate_pruned_space(256, 256, 256, mode='double')
    outside_pool = []
    for bi in [1, 2, 4, 8, 16]:
        for bj in [1, 2, 4, 8, 16]:
            ti = 256 // bi; tj = 256 // bj
            if ti * tj < 32: continue
            for bk in get_divisors(256)[:5]:
                if not check_single_pruning(bi, bj, bk, ti*tj):
                    outside_pool.append((bi, bj, bk, ti*tj))

    import random; random.seed(42)
    outside_pool = random.sample(outside_pool, min(n_per_group, len(outside_pool)))

    results = {'outside': [], 'single': [], 'double': []}

    def test_pool(pool, label):
        print(f'\n--- Testing {label} ({len(pool)} configs) ---')
        for idx, (bi, bj, bk, threads) in enumerate(pool):
            ti = 256 // bi; tj = 256 // bj
            rk = 256 // bk if bk > 0 else 256

            # Build modified decisions
            new_dec = []
            for dec in decisions:
                if dec[0] == sp_idx[0]:
                    new_dec.append([dec[0], [bi, 1, ti, 1, 1]])
                elif dec[0] == sp_idx[1]:
                    new_dec.append([dec[0], [bj, 1, tj, 1, 1]])
                elif dec[0] == rd_idx[0]:
                    new_dec.append([dec[0], [bk, 1, rk]])
                else:
                    new_dec.append(dec)

            mod_data = [instructions, new_dec]
            tvm_result = 'unknown'
            try:
                sch = Schedule(mod, debug_mask='all')
                TraceApplyJSONToSchedule(mod_data, sch)
                ok, bad = apply_postprocs(sch, pps)
                tvm_result = 'PASS' if ok else bad
            except Exception as e:
                tvm_result = f'setup:{type(e).__name__}'

            single_pred = check_single_pruning(bi, bj, bk, threads)
            double_pred = check_double_pruning(bi, bj, bk, threads)
            entry = {'bi': bi, 'bj': bj, 'bk': bk, 'threads': threads,
                     'single_pred': single_pred, 'double_pred': double_pred,
                     'tvm_result': tvm_result, 'tvm_pass': tvm_result == 'PASS'}

            results[label].append(entry)
            mark = 'Y' if tvm_result == 'PASS' else 'N'
            print(f'  [{idx+1:2d}/{len(pool)}] ({bi:3d},{bj:3d},{bk:3d}) thr={threads:4d} '
                  f's={single_pred} d={double_pred} TVM={mark} {tvm_result}')

    test_pool(outside_pool, 'outside')
    test_pool(single_pool[:n_per_group], 'single')
    test_pool(double_pool[:n_per_group], 'double')

    # Summary
    print(f'\n{"="*60}')
    for label in ['outside', 'single', 'double']:
        pool = results[label]
        n = len(pool)
        n_pass = sum(1 for r in pool if r['tvm_pass'])
        rate = n_pass / n * 100 if n > 0 else 0
        print(f'  {label:10s}: {n:3d} tested, {n_pass:3d} passed ({rate:.1f}%)')

    return results


def get_divisors(n):
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0: divs.add(i); divs.add(n // i)
    return sorted(divs)


if __name__ == '__main__':
    results = run(n_per_group=10)
    od = 'F:/tvm/experiments/data'
    os.makedirs(od, exist_ok=True)
    with open(f'{od}/exp02_verify.json', 'w') as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    print(f'\nSaved: {od}/exp02_verify.json')
