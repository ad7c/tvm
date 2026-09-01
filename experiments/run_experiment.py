"""
Run experiment in your terminal. Shows progress every attempt.
Usage: python experiments/run_experiment.py
"""
import sys, os, time, random, json, math
from collections import Counter

os.environ['USERPROFILE'] = 'C:\\tvm_home'
os.environ['HOME'] = 'C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH'] = 'F:\\tvm\\build\\lib'
sys.path.insert(0, 'F:\\tvm\\python')

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


# ═══ MatMul 128x128x128 ═══
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

OUTPUT_DIR = 'F:/tvm/experiments/data'

# ═══ Tile extraction + Prefilter ═══
def extract(info_dict, trace):
    """Return (bi, bj, bk, threads, vk) or None."""
    dm = trace.decisions; sp, rd = [], []
    for k, v in list(dm.items()):
        if 'sample_perfect_tile' not in str(k).lower(): continue
        try: vs = tuple(int(vv.value) if hasattr(vv, 'value') else int(vv) for vv in list(v))
        except: continue
        (sp if 'n=5' in str(k) else rd).append(vs)
    if len(sp) < 2: return None
    si, sj = sp[0], sp[1]
    bi, ti = si[0], si[2]; bj, tj = sj[0], sj[2]
    bk = rd[0][0] if rd else 0; threads = ti * tj
    vk = rd[0][1] if rd else None
    info_dict['bi'], info_dict['bj'], info_dict['bk'] = bi, bj, bk
    info_dict['threads'], info_dict['vk'] = threads, vk
    return bi, bj, bk, threads, vk

def est_shmem(bi, bj, bk): return (bi*bk + bk*bj)*8 + bi*bj*4

def prefilter(info_dict):
    b, t, vk = info_dict['bi'], info_dict['threads'], info_dict['vk']
    if b is None: return False, 'no_data'
    if t and (t < 32 or t > 1024): return False, f'threads={t}'
    if b > 0:
        sh = est_shmem(b, info_dict.get('bj',0), info_dict.get('bk',0))
        if sh > GPU_SHMEM: return False, f'shmem={sh}'
    if vk is not None and vk == 1: return False, f'vk=1'
    return True, 'ok'

# ═══ One mode ═══
def run_mode(label, use_prefilter, max_attempts, n_needed):
    sg = PostOrderApply()
    ctx = TuneContext(mod=Matmul128, target=TARGET, space_generator=sg, task_name='run')
    ctx.space_generator._initialize_with_tune_context(ctx)
    ds = ctx.space_generator.generate_design_space(Matmul128)
    insts = [d.trace.insts for d in ds]

    postprocs = Postproc.create('cuda')
    skip = {'RewriteTensorize', 'RewriteUnboundBlock'}
    pps = [pp for pp in postprocs if type(pp).__name__ not in skip]

    print(f'\n{"="*60}')
    print(f'{label}')
    print(f'  Prefilter: {"ON" if use_prefilter else "OFF (baseline)"}')
    print(f'  Target: {n_needed} valid, max {max_attempts} attempts')
    print(f'  Postproc: {" -> ".join(type(p).__name__ for p in pps)}')
    print(f'{"="*60}')

    random.seed(42)
    valid_at, valid_info = [], []
    stats = Counter()
    t_start = time.time()

    for attempt in range(1, max_attempts + 1):
        t = Trace(insts[random.randint(0, 2)], {})
        info_dict = {}
        try:
            sch = Schedule(Matmul128, debug_mask='all')
            t.apply_to_schedule(sch, remove_postproc=False)

            # Prefilter
            pruned = False
            if use_prefilter:
                extract(info_dict, sch.trace)
                ok, why = prefilter(info_dict)
                if not ok:
                    stats['pruned'] += 1
                    pruned = True

            if not pruned:
                sch.enter_postproc()
                ok = True
                for pp in pps:
                    if not pp.apply(sch):
                        ok = False
                        stats[f'tvm_{type(pp).__name__}'] += 1
                        break
                if ok:
                    extract(info_dict, sch.trace)
                    valid_at.append(attempt)
                    valid_info.append(dict(info_dict))
                    stats['passed'] += 1

        except Exception as e:
            stats[f'exc_{type(e).__name__}'] += 1

        # === PROGRESS BAR (every attempt) ===
        if attempt % 10 == 0 or attempt <= 10 or len(valid_at) > len(valid_info) - 1:
            elapsed = time.time() - t_start
            bar_len = 40
            filled = int(bar_len * attempt / max_attempts)
            bar = '#' * filled + '-' * (bar_len - filled)
            n_pruned = stats.get('pruned', 0)
            tv = attempt - n_pruned
            print(f'\r  [{attempt:5d}/{max_attempts}] [{bar}] valid={len(valid_at)} '
                  f'pruned={n_pruned} tvm_calls={tv} '
                  f'{elapsed:.0f}s {"PASS" if pruned==False and stats.get("passed",0)>len(valid_info)-1 else ""}',
                  end='', flush=True)

        if len(valid_at) >= n_needed:
            break

    elapsed = time.time() - t_start
    print()  # newline after progress bar
    n_pruned = stats.get('pruned', 0)

    print(f'\n  DONE: {attempt} attempts, {len(valid_at)} valid, {elapsed:.0f}s')
    print(f'  Pruned: {n_pruned} ({n_pruned/attempt*100:.1f}%), TVM calls: {attempt-n_pruned-len(valid_at)}')
    if valid_at:
        print(f'  Valid at attempts: {valid_at}')
        print(f'  First valid: attempt {valid_at[0]}')
        for i, vi in enumerate(valid_info):
            print(f'    #{i+1}: bi={vi["bi"]} bj={vi["bj"]} bk={vi["bk"]} threads={vi["threads"]} vk={vi["vk"]}')

    return {'attempts': attempt, 'valid': len(valid_at), 'valid_at': valid_at,
            'valid_info': valid_info, 'time_s': elapsed, 'pruned': n_pruned,
            'tvm_calls': attempt - n_pruned - len(valid_at), 'stats': dict(stats)}


# ═══ Main ═══
if __name__ == '__main__':
    SESSIONS = 3          # Run 3 sessions for averaging
    MAX_ATTEMPTS = 3000   # Per mode per session
    NEED_VALID = 3        # Valid candidates needed per mode per session

    all_sessions = []

    for sess in range(1, SESSIONS + 1):
        print(f'\n{"#"*70}')
        print(f'# SESSION {sess}/{SESSIONS}')
        print(f'{"#"*70}')

        session_results = {}
        for mode, prefilter_flag, desc in [
            ('baseline', False, 'BASELINE (no prefilter)'),
            ('single', True, 'SINGLE (upward: shmem+threads+vk!=1)'),
        ]:
            session_results[mode] = run_mode(
                f'S{sess} {desc}', prefilter_flag, MAX_ATTEMPTS, NEED_VALID)

        all_sessions.append(session_results)

        # Save checkpoint
        with open(f'{OUTPUT_DIR}/experiment_s{sess}.json', 'w') as f:
            json.dump(session_results, f, indent=2, default=str)
        print(f'  Saved: {OUTPUT_DIR}/experiment_s{sess}.json')

    # Final summary
    print(f'\n{"="*70}')
    print(f'FINAL SUMMARY ({SESSIONS} sessions)')
    print(f'{"="*70}')

    for mode in ['baseline', 'single']:
        times = [s[mode]['time_s'] for s in all_sessions]
        attempts = [s[mode]['attempts'] for s in all_sessions]
        valids = [s[mode]['valid'] for s in all_sessions]
        pruned = [s[mode]['pruned'] for s in all_sessions]
        tvms = [s[mode]['tvm_calls'] for s in all_sessions]

        avg_time = sum(times) / len(times)
        avg_valid = sum(valids) / len(valids)
        avg_attempts = sum(attempts) / len(attempts)

        print(f'\n  {mode}:')
        print(f'    Avg time: {avg_time:.0f}s, Avg attempts: {avg_attempts:.0f}')
        print(f'    Avg valid: {avg_valid:.1f}, Avg pruned: {sum(pruned)/len(pruned):.0f}')
        if all(v > 0 for v in valids):
            print(f'    Avg attempts/valid: {sum(attempts)/sum(valids):.0f}')

    print(f'\nAll data saved to: {OUTPUT_DIR}/experiment_s*.json')
