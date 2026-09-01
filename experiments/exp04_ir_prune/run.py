"""
Exp 04: IR-level pruning — check actual shared_memory buffer size
after RewriteCooperativeFetch, before VerifyGPUCode.
"""
import sys, os, time, random, json, re
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


def get_real_shmem_usage(sch):
    """Scan the schedule's IR for actual shared memory allocations."""
    ir = sch.mod.script()
    total = 0
    for m in re.finditer(r'alloc_buffer\(\((\d+),\s*(\d+)\).*shared', ir):
        total += int(m.group(1)) * int(m.group(2)) * 4
    for m in re.finditer(r'alloc_buffer\(\((\d+),\s*(\d+),\s*(\d+)\).*shared', ir):
        total += int(m.group(1)) * int(m.group(2)) * int(m.group(3)) * 4
    return total


def extract_tile_params(trace):
    dm = trace.decisions; sp, rd = [], []
    for key, val in list(dm.items()):
        if 'sample_perfect_tile' not in str(key).lower(): continue
        try:
            vs = tuple(int(v.value) if hasattr(v, 'value') else int(v) for v in list(val))
        except: continue
        if 'n=5' in str(key): sp.append(vs)
        elif 'n=3' in str(key): rd.append(vs)
    if len(sp) < 2: return None, None, None, None
    si, sj = sp[0], sp[1]; bi, ti = si[0], si[2]; bj, tj = sj[0], sj[2]
    bk = rd[0][0] if rd else None; threads = ti * tj
    return bi, bj, bk, threads


def prefilter_pass(bi, bj, bk, threads):
    """Analytical pre-filter (same as exp03)."""
    if bi is None: return False
    if threads is not None and (threads < 32 or threads > 1024): return False
    if bk is not None and bj is not None:
        shmem = (bi*bk + bk*bj)*8 + bi*bj*4
        if shmem > GPU_SHMEM: return False
    return True


def run_experiment(label, mode, max_attempts, n_valid_needed):
    """
    mode: 'none' = baseline, 'analytical' = exp03 prefilter,
          'ir' = IR-level shmem check after CoopFetch
    """
    sg = PostOrderApply()
    ctx = TuneContext(mod=Matmul128, target=TARGET, space_generator=sg, task_name='exp04')
    ctx.space_generator._initialize_with_tune_context(ctx)
    ds_list = ctx.space_generator.generate_design_space(Matmul128)
    trace_insts = [ds.trace.insts for ds in ds_list]

    postprocs = Postproc.create('cuda')
    skip = {'RewriteTensorize', 'RewriteUnboundBlock'}
    pps = [pp for pp in postprocs if type(pp).__name__ not in skip]

    print(f'\n{"="*60}')
    print(f'{label}: mode={mode}')
    print(f'Target: {n_valid_needed} valid, max {max_attempts} attempts')
    print(f'{"="*60}')

    valid_found = 0; valid_at = []; stats = Counter()
    t0 = time.time()

    for attempt in range(1, max_attempts + 1):
        ds_idx = random.randint(0, 2)
        trace = Trace(trace_insts[ds_idx], {})

        try:
            sch = Schedule(Matmul128, debug_mask='all')
            trace.apply_to_schedule(sch, remove_postproc=True)

            # ── Analytical prefilter (cheapest) ──
            if mode != 'none':
                bi, bj, bk, threads = extract_tile_params(sch.trace)
                if not prefilter_pass(bi, bj, bk, threads):
                    stats['ana_pruned'] += 1
                    continue
                stats['ana_pass'] += 1

            sch.enter_postproc()
            ok = True; bad = None; ir_pruned = False

            for pp in pps:
                pp_name = type(pp).__name__
                if not pp.apply(sch):
                    ok = False; bad = pp_name; break

                # ── IR-level prune: after CoopFetch, check real shmem ──
                if mode == 'ir' and pp_name == 'RewriteCooperativeFetch':
                    real_shmem = get_real_shmem_usage(sch)
                    if real_shmem > GPU_SHMEM:
                        ir_pruned = True
                        stats['ir_pruned'] += 1
                        break

            if ir_pruned:
                continue

            if ok:
                valid_found += 1; valid_at.append(attempt)
                bi, bj, bk, threads = extract_tile_params(sch.trace)
                real_shmem = get_real_shmem_usage(sch)
                print(f'  [{attempt:5d}] VALID #{valid_found}! bi={bi}, bj={bj}, bk={bk}, '
                      f'threads={threads}, real_shmem={real_shmem}', flush=True)
                stats['passed'] += 1
                if valid_found >= n_valid_needed: break
            else:
                stats[f'tvm_{bad or "?"}'] += 1

        except Exception as e:
            stats[f'exc_{type(e).__name__}'] += 1

        if attempt % 300 == 0:
            elapsed = time.time() - t0
            print(f'  [{attempt:5d}] {elapsed:.0f}s valid={valid_found} stats={dict(stats)}', flush=True)

    elapsed = time.time() - t0
    ana_pruned = stats.get('ana_pruned', 0)
    ir_pruned = stats.get('ir_pruned', 0)
    total_pruned = ana_pruned + ir_pruned
    tvm_tested = attempt - total_pruned - valid_found

    print(f'\n--- {label} RESULTS ---')
    print(f'  Attempts: {attempt}, Valid: {valid_found}')
    print(f'  Elapsed: {elapsed:.0f}s')
    print(f'  Pruned: ana={ana_pruned} ir={ir_pruned} total={total_pruned} ({total_pruned/attempt*100:.1f}%)')
    print(f'  TVM tested: {tvm_tested}')
    if valid_found:
        print(f'  Valid at: {valid_at}')
    print(f'  Breakdown: {dict(stats)}')

    return {'label': label, 'mode': mode, 'attempts': attempt, 'valid': valid_found,
            'valid_at': valid_at, 'elapsed_s': elapsed, 'ana_pruned': ana_pruned,
            'ir_pruned': ir_pruned, 'total_pruned': total_pruned,
            'tvm_tested': tvm_tested, 'stats': dict(stats)}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='both', choices=['baseline', 'analytical', 'ir', 'both'])
    args = ap.parse_args()

    MAX = 3000; NEED = 3; random.seed(42)
    results = {}

    if args.mode in ('baseline', 'both'):
        r = run_experiment('BASELINE', 'none', MAX, NEED)
        results['baseline'] = r

    if args.mode in ('analytical', 'both'):
        r = run_experiment('ANALYTICAL', 'analytical', MAX, NEED)
        results['analytical'] = r

    if args.mode in ('ir', 'both'):
        r = run_experiment('IR_PRUNE', 'ir', MAX, NEED)
        results['ir'] = r

    # Save
    od = 'F:/tvm/experiments/data'
    os.makedirs(od, exist_ok=True)
    with open(f'{od}/exp04_ir_prune.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: {od}/exp04_ir_prune.json')
