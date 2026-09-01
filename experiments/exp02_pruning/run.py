"""
Exp 02: Bidirectional pruning — pre-filter candidates before TVM postproc.
Compare with exp01 baseline: how many attempts to find first valid?
"""
import sys, os, time, random, json
from collections import Counter
os.environ['USERPROFILE'] = 'C:\\tvm_home'
os.environ['HOME'] = 'C:\\tvm_home'
sys.path.insert(0, 'F:/tvm/python')
os.environ['TVM_LIBRARY_PATH'] = 'F:/tvm/build/lib'
sys.path.insert(0, 'F:/tvm/experiments')

from pruning import check_upward, check_downward, extract_tile_params

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


def run(max_attempts=1000, mode='upward'):
    """
    mode: 'none' (baseline), 'upward' (hardware only), 'double' (up+down)
    """
    # Design spaces
    sg = PostOrderApply()
    ctx = TuneContext(mod=Matmul128, target=TARGET, space_generator=sg, task_name='exp02')
    ctx.space_generator._initialize_with_tune_context(ctx)
    ds_list = ctx.space_generator.generate_design_space(Matmul128)
    trace_insts = [ds.trace.insts for ds in ds_list]

    # Postproc
    postprocs = Postproc.create('cuda')
    skip = {'RewriteTensorize', 'RewriteUnboundBlock'}
    pps = [pp for pp in postprocs if type(pp).__name__ not in skip]

    print(f'Mode: {mode}')
    print(f'Design spaces: {len(trace_insts)}, Postproc: {len(pps)} stages')
    print(f'Sampling until 1st valid or {max_attempts} attempts...\n')

    stats = Counter()
    first_valid = None
    tile_info = None

    t0 = time.time()
    random.seed(42)

    for attempt in range(1, max_attempts + 1):
        ds_idx = random.randint(0, len(trace_insts) - 1)
        trace = Trace(trace_insts[ds_idx], {})

        # === Generate schedule (replay trace once) ===
        bi = bj = bk = threads = None
        pre_ok = True
        pre_reason = 'no_check'

        try:
            sch = Schedule(Matmul128, debug_mask='all')
            trace.apply_to_schedule(sch, remove_postproc=True)

            if mode != 'none':
                bi, bj, bk, threads = extract_tile_params(sch.trace)

                if mode == 'upward':
                    pre_ok, pre_reason = check_upward(bi, bj, bk, threads)
                elif mode == 'double':
                    up_ok, up_reason = check_upward(bi, bj, bk, threads)
                    if up_ok:
                        pre_ok, pre_reason = check_downward(bi, bj, bk, threads)
                    else:
                        pre_ok, pre_reason = False, up_reason

            if not pre_ok:
                stats[f'prune_{pre_reason}'] += 1
                continue

            # === TVM postproc ===
            sch.enter_postproc()
            ok = True
            bad = None
            for pp in pps:
                pp_name = type(pp).__name__
                if not pp.apply(sch):
                    ok = False
                    bad = pp_name
                    break
        except Exception as e:
            bad = f'({type(e).__name__})'
            ok = False

        if ok:
            first_valid = attempt
            tile_info = {'bi': bi, 'bj': bj, 'bk': bk, 'threads': threads}
            print(f'\n  *** FIRST VALID at attempt {attempt}! ***')
            print(f'      bi={bi}, bj={bj}, bk={bk}, threads={threads}')
            break
        else:
            stats[f'tvm_{bad or "?"}'] += 1

        if attempt % 200 == 0:
            elapsed = time.time() - t0
            total = attempt
            tvm_tested = sum(v for k, v in stats.items() if k.startswith('tvm_'))
            prune_skipped = sum(v for k, v in stats.items() if k.startswith('prune_'))
            print(f'  [{attempt:4d}] pruned={prune_skipped} tvm_tested={tvm_tested} '
                  f'pass=0 | {elapsed:.0f}s')

    elapsed = time.time() - t0

    # Summary
    total_attempts = attempt
    pruned = sum(v for k, v in stats.items() if k.startswith('prune_'))
    tvm_fails = sum(v for k, v in stats.items() if k.startswith('tvm_'))

    print(f'\n{"="*60}')
    print(f'RESULTS ({mode} pruning): {total_attempts} attempts, {elapsed:.0f}s')
    print(f'  Candidates pruned (avoided TVM): {pruned} ({pruned/total_attempts*100:.1f}%)')
    print(f'  Candidates sent to TVM:          {tvm_fails + (1 if first_valid else 0)}')
    if first_valid:
        print(f'  First valid at attempt:          {first_valid}')
        print(f'  Effective rate (overall):        {1/total_attempts*100:.2f}%')
        print(f'  Tile: bi={tile_info["bi"]}, bj={tile_info["bj"]}, '
              f'bk={tile_info["bk"]}, threads={tile_info["threads"]}')
    else:
        print(f'  No valid candidate in {max_attempts} attempts')
    print(f'  Breakdown:')
    for k, v in stats.most_common():
        print(f'    {k}: {v}')

    return {
        'mode': mode,
        'attempts': total_attempts,
        'first_valid_at': first_valid,
        'pruned': pruned,
        'tvm_tested': tvm_fails + (1 if first_valid else 0),
        'tile': tile_info,
        'stats': dict(stats),
        'elapsed_s': elapsed,
    }


if __name__ == '__main__':
    # Run pruning modes only (baseline from exp01)
    results = {}
    for mode in ['upward', 'double']:
        print(f'\n{"="*60}')
        print(f'  MODE: {mode}')
        print(f'{"="*60}')
        r = run(1000, mode=mode)
        results[mode] = r

    # Save
    od = 'F:/tvm/experiments/data'
    os.makedirs(od, exist_ok=True)
    with open(f'{od}/exp02_pruning.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: {od}/exp02_pruning.json')
