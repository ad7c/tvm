"""
Exp 06: Three-way comparison — baseline vs single pruning vs double pruning.
Double pruning adds occupancy + data reuse (no warp alignment, no block%warp).
"""
import sys, os, time, random, json
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

GPU = {'shmem': 49152, 'threads': 1024, 'threads_sm': 1536, 'warp': 32}


def extract_tile_params(trace):
    dm = trace.decisions; sp, rd = [], []
    for key, val in list(dm.items()):
        if 'sample_perfect_tile' not in str(key).lower(): continue
        try: vs = tuple(int(v.value) if hasattr(v,'value') else int(v) for v in list(val))
        except: continue
        if 'n=5' in str(key): sp.append(vs)
        elif 'n=3' in str(key): rd.append(vs)
    if len(sp) < 2: return None,None,None,None,None,None
    si, sj = sp[0], sp[1]; bi, ti = si[0], si[2]; bj, tj = sj[0], sj[2]
    bk = rd[0][0] if rd else None; threads = ti*tj
    return bi, bj, bk, threads, ti, tj


def est_shmem(bi, bj, bk): return (bi*bk + bk*bj)*8 + bi*bj*4

def occupancy(threads, shmem):
    b_threads = GPU['threads_sm'] // max(threads, 1)
    b_shmem = 102400 // max(shmem, 1)
    blocks = max(1, min(b_threads, b_shmem))
    return blocks * threads / GPU['threads_sm']


def prefilter_single(bi, bj, bk, threads):
    """Upward: threads + shmem."""
    if bi is None: return False
    if threads and (threads < 32 or threads > 1024): return False
    if bk and bj and est_shmem(bi,bj,bk) > GPU['shmem']: return False
    return True


def prefilter_double(bi, bj, bk, threads):
    """Upward + downward: occupancy + data reuse."""
    if not prefilter_single(bi, bj, bk, threads): return False
    if bk and bj:
        shmem = est_shmem(bi,bj,bk)
        if shmem > 0 and occupancy(threads, shmem) < 0.25: return False
        flops = 2*bi*bj*bk; mem = (bi*bk + bk*bj)*4
        if mem > 0 and flops/mem < 8.0: return False
    return True


def run_experiment(label, max_attempts, n_needed, prefilter=None):
    """prefilter: None=baseline, 'single', 'double'"""
    sg = PostOrderApply()
    ctx = TuneContext(mod=Matmul128, target=TARGET, space_generator=sg, task_name='exp06')
    ctx.space_generator._initialize_with_tune_context(ctx)
    ds_list = ctx.space_generator.generate_design_space(Matmul128)
    insts = [ds.trace.insts for ds in ds_list]

    postprocs = Postproc.create('cuda')
    skip = {'RewriteTensorize', 'RewriteUnboundBlock'}
    pps = [pp for pp in postprocs if type(pp).__name__ not in skip]

    print(f'\n{"="*60}')
    pf_name = {'single': 'SINGLE (upward)', 'double': 'DOUBLE (up+down)', None: 'BASELINE (none)'}[prefilter]
    print(f'{label}: {pf_name}')
    print(f'Target: {n_needed} valid, max {max_attempts} attempts')
    print(f'{"="*60}')

    valid_at = []; stats = Counter(); t0 = time.time()
    random.seed(42)

    for attempt in range(1, max_attempts+1):
        ds_idx = random.randint(0, 2)
        trace = Trace(insts[ds_idx], {})

        try:
            sch = Schedule(Matmul128, debug_mask='all')
            trace.apply_to_schedule(sch, remove_postproc=True)

            # === Prefilter ===
            if prefilter:
                bi, bj, bk, threads, ti, tj = extract_tile_params(sch.trace)
                if prefilter == 'single':
                    if not prefilter_single(bi, bj, bk, threads):
                        stats['pruned'] += 1; continue
                elif prefilter == 'double':
                    if not prefilter_double(bi, bj, bk, threads):
                        stats['pruned'] += 1; continue

            sch.enter_postproc()
            ok = True; bad = None
            for pp in pps:
                if not pp.apply(sch):
                    ok = False; bad = type(pp).__name__; break

            if ok:
                valid_at.append(attempt)
                print(f'  [{attempt:5d}] VALID #{len(valid_at)}! ({time.time()-t0:.0f}s)',
                      flush=True)
                stats['passed'] += 1
                if len(valid_at) >= n_needed: break
            else:
                stats[f'tvm_{bad}'] += 1
        except Exception as e:
            stats[f'exc_{type(e).__name__}'] += 1

        if attempt % 300 == 0:
            elapsed = time.time()-t0
            n_pruned = stats.get('pruned', 0)
            print(f'  [{attempt:5d}] {elapsed:.0f}s valid={len(valid_at)} '
                  f'pruned={n_pruned}', flush=True)

    elapsed = time.time()-t0
    total_pruned = stats.get('pruned', 0)
    tvm_calls = attempt - total_pruned - len(valid_at)

    print(f'\n--- {label} ({pf_name}) ---')
    print(f'  Attempts: {attempt}, Valid: {len(valid_at)}, Time: {elapsed:.0f}s')
    print(f'  Pruned: {total_pruned} ({total_pruned/attempt*100:.1f}%)')
    print(f'  TVM calls: {tvm_calls}')
    if valid_at: print(f'  Valid at: {valid_at}')
    print(f'  Breakdown: {dict(stats.most_common())}')

    return {'attempts': attempt, 'valid': len(valid_at), 'time_s': elapsed,
            'pruned': total_pruned, 'tvm_calls': tvm_calls, 'valid_at': valid_at,
            'stats': dict(stats)}


if __name__ == '__main__':
    MAX, NEED = 3000, 3
    results = {}

    for mode, label in [('baseline', 'BASELINE'), ('single', 'SINGLE'), ('double', 'DOUBLE')]:
        results[label.lower()] = run_experiment(
            label, max_attempts=MAX, n_needed=NEED,
            prefilter=None if mode == 'baseline' else mode
        )

    # Summary
    print(f'\n{"="*60}')
    print(f'COMPARISON')
    print(f'{"="*60}')
    h = f'  {"Metric":25s} {"Baseline":>10s} {"Single":>10s} {"Double":>10s}'
    print(h); print('-'*60)
    keys = ['attempts', 'valid', 'time_s', 'tvm_calls', 'pruned']
    names = {'attempts': 'Total attempts', 'valid': 'Valid found', 'time_s': 'Time (s)',
             'tvm_calls': 'TVM calls', 'pruned': 'Pruned'}
    for k in keys:
        vals = [str(results[m][k]) for m in ['baseline','single','double']]
        print(f'  {names[k]:25s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}')

    # Save
    od = 'F:/tvm/experiments/data'
    os.makedirs(od, exist_ok=True)
    with open(f'{od}/exp06_double.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: {od}/exp06_double.json')
