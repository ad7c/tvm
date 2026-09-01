"""
Exp 03: Verify pruning effectiveness — controlled comparison.
  Baseline: random sampling -> postproc, find 3 valid
  Pruning:  random sampling -> pre-filter (shmem+threads) -> postproc, find 3 valid
Compare: attempts saved, time saved.
"""
import sys, os, time, random, json, math
from collections import Counter

os.environ['USERPROFILE'] = 'C:\\tvm_home'
os.environ['HOME'] = 'C:\\tvm_home'
sys.path.insert(0, 'F:/tvm/python')
os.environ['TVM_LIBRARY_PATH'] = 'F:/tvm/build/lib'

from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build, default_export
register_global_func('default')(default_build)
register_global_func('default_export')(default_export)

import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.schedule import Schedule
from tvm.s_tir.schedule.trace import Trace
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply
from tvm.s_tir.meta_schedule.postproc import Postproc


@tvm.script.ir_module
class Matmul128:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({'global_symbol': 'main'})
        A = T.match_buffer(a, (128, 128), 'float32')
        B = T.match_buffer(b, (128, 128), 'float32')
        C = T.match_buffer(c, (128, 128), 'float32')
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock('matmul'):
                vi, vj, vk = T.axis.remap('SSR', [i, j, k])
                with T.init(): C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


TARGET = Target({
    'kind': 'cuda', 'arch': 'sm_86',
    'max_threads_per_block': 1024, 'max_num_threads': 1024,
    'max_shared_memory_per_block': 49152, 'thread_warp_size': 32,
})

GPU_SHMEM = 49152
WARP = 32
MAX_THREADS = 1024


# ── Pre-filter ──

def extract_tile_params(trace):
    dm = trace.decisions
    sp, rd = [], []
    for key, val in list(dm.items()):
        if 'sample_perfect_tile' not in str(key).lower():
            continue
        try:
            vs = tuple(int(v.value) if hasattr(v, 'value') else int(v) for v in list(val))
        except:
            continue
        if 'n=5' in str(key): sp.append(vs)
        elif 'n=3' in str(key): rd.append(vs)
    if len(sp) < 2: return None, None, None, None
    si, sj = sp[0], sp[1]
    bi, ti = si[0], si[2]
    bj, tj = sj[0], sj[2]
    bk = rd[0][0] if rd else None
    threads = ti * tj
    return bi, bj, bk, threads


def estimate_shmem(bi, bj, bk):
    return (bi * bk + bk * bj) * 4 * 2 + bi * bj * 4


def prefilter_pass(bi, bj, bk, threads):
    """Only 2 rules: threads in range, shmem under limit."""
    if bi is None: return False
    if threads is not None:
        if threads < WARP or threads > MAX_THREADS:
            return False
    if bk is not None and bj is not None:
        if estimate_shmem(bi, bj, bk) > GPU_SHMEM:
            return False
    return True


# ── Runner ──

def run_experiment(label, use_prefilter, max_attempts, n_valid_needed):
    sg = PostOrderApply()
    ctx = TuneContext(mod=Matmul128, target=TARGET, space_generator=sg, task_name='exp03')
    ctx.space_generator._initialize_with_tune_context(ctx)
    ds_list = ctx.space_generator.generate_design_space(Matmul128)
    trace_insts = [ds.trace.insts for ds in ds_list]

    postprocs = Postproc.create('cuda')
    skip = {'RewriteTensorize', 'RewriteUnboundBlock'}
    pps = [pp for pp in postprocs if type(pp).__name__ not in skip]

    print(f'\n{"="*60}')
    print(f'{label}: {"prefilter ON" if use_prefilter else "baseline (no prefilter)"}')
    print(f'Target: {n_valid_needed} valid candidates, max {max_attempts} attempts')
    print(f'{"="*60}')

    valid_found = 0
    valid_at = []
    stats = Counter()
    t0 = time.time()

    for attempt in range(1, max_attempts + 1):
        ds_idx = random.randint(0, 2)
        trace = Trace(trace_insts[ds_idx], {})

        ok = False
        bad = None

        try:
            sch = Schedule(Matmul128, debug_mask='all')
            trace.apply_to_schedule(sch, remove_postproc=True)

            # Pre-filter
            if use_prefilter:
                bi, bj, bk, threads = extract_tile_params(sch.trace)
                if not prefilter_pass(bi, bj, bk, threads):
                    stats['pruned'] += 1
                    continue

            sch.enter_postproc()
            ok = True
            for pp in pps:
                pp_name = type(pp).__name__
                if not pp.apply(sch):
                    ok = False
                    bad = pp_name
                    break
        except Exception as e:
            bad = f'({type(e).__name__})'

        if ok:
            valid_found += 1
            valid_at.append(attempt)
            bi, bj, bk, threads = extract_tile_params(sch.trace)
            print(f'  [{attempt:5d}] VALID #{valid_found}! bi={bi}, bj={bj}, bk={bk}, threads={threads}, est_shmem={estimate_shmem(bi,bj,bk)}')
            stats['passed'] += 1
            if valid_found >= n_valid_needed:
                break
        else:
            stats[f'tvm_{bad or "?"}'] += 1

        if attempt % 200 == 0:
            elapsed = time.time() - t0
            pruned = stats.get('pruned', 0)
            tvm_tested = attempt - pruned
            s = f'  [{attempt:5d}] elapsed={elapsed:.0f}s valid={valid_found} pruned={pruned} tvm_tested={tvm_tested}'
            print(s, flush=True)

    elapsed = time.time() - t0
    total_attempts = attempt
    pruned = stats.get('pruned', 0)

    print(f'\n--- {label} RESULTS ---')
    print(f'  Attempts: {total_attempts}, Valid: {valid_found}')
    print(f'  Elapsed: {elapsed:.0f}s')
    print(f'  Valid at attempts: {valid_at}')
    if pruned > 0:
        print(f'  Pruned (avoided TVM): {pruned} ({pruned/total_attempts*100:.1f}%)')
    if valid_found > 0:
        gaps = [valid_at[0]] + [valid_at[i] - valid_at[i-1] for i in range(1, len(valid_at))]
        print(f'  Gaps between valids: {gaps}')
    print(f'  Breakdown: {dict(stats)}')

    return {
        'label': label,
        'use_prefilter': use_prefilter,
        'total_attempts': total_attempts,
        'valid_found': valid_found,
        'valid_at': valid_at,
        'elapsed_s': elapsed,
        'pruned': pruned,
        'stats': dict(stats),
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='both', choices=['baseline', 'pruning', 'both'])
    args = ap.parse_args()

    MAX = 3000
    NEED = 3
    random.seed(42)

    results = {}

    if args.mode in ('baseline', 'both'):
        r1 = run_experiment('BASELINE', use_prefilter=False, max_attempts=MAX, n_valid_needed=NEED)
        results['baseline'] = r1

    if args.mode in ('pruning', 'both'):
        r2 = run_experiment('PRUNING', use_prefilter=True, max_attempts=MAX, n_valid_needed=NEED)
        results['pruning'] = r2

    # Summary table
    if 'baseline' in results and 'pruning' in results:
        r1 = results['baseline']
        r2 = results['pruning']
        print(f'\n{"="*60}')
        print(f'COMPARISON')
        print(f'{"="*60}')
        print(f'  {"Metric":30s} {"Baseline":>12s} {"Pruning":>12s}')
        print(f'  {"-"*56}')
        print(f'  {"Total attempts":30s} {r1["total_attempts"]:>12,} {r2["total_attempts"]:>12,}')
        if r1['valid_found'] > 0 and r2['valid_found'] > 0:
            avg1 = r1['total_attempts'] / r1['valid_found']
            avg2 = r2['total_attempts'] / r2['valid_found']
            print(f'  {"Avg attempts/valid":30s} {avg1:>12.0f} {avg2:>12.0f}')
        print(f'  {"Elapsed (s)":30s} {r1["elapsed_s"]:>12.0f} {r2["elapsed_s"]:>12.0f}')
        if r2['pruned'] > 0:
            saved = r2['pruned']
            tvm_calls_baseline = r1['total_attempts']
            tvm_calls_pruning = r2['total_attempts'] - r2['pruned']
            print(f'  {"TVM postproc calls":30s} {tvm_calls_baseline:>12,} {tvm_calls_pruning:>12,}')
            print(f'  {"TVM calls saved":30s} {"-":>12s} {saved:>12,}')
        print(f'  {"First valid at":30s} {r1["valid_at"][0] if r1["valid_at"] else "N/A":>12} {r2["valid_at"][0] if r2["valid_at"] else "N/A":>12}')

    # Save
    od = 'F:/tvm/experiments/data'
    os.makedirs(od, exist_ok=True)
    with open(f'{od}/exp03_verify.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: {od}/exp03_verify.json')
