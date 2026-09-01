"""
Exp 01: Meta Schedule random sampling on MatMul 128^3.
Count attempts to find valid candidates (same logic as C++ SampleInitPopulation).
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


def run(max_attempts=1000):
    # Step 1: Generate design spaces (trace templates)
    sg = PostOrderApply()
    ctx = TuneContext(mod=Matmul128, target=TARGET, space_generator=sg, task_name='exp01')
    ctx.space_generator._initialize_with_tune_context(ctx)
    ds_list = ctx.space_generator.generate_design_space(Matmul128)

    # Extract instruction templates
    trace_insts = [ds.trace.insts for ds in ds_list]
    print(f'Design spaces: {len(trace_insts)}')
    for i, insts in enumerate(trace_insts):
        print(f'  DS[{i}]: {len(insts)} instructions')

    # Postproc chain
    postprocs = Postproc.create('cuda')
    skip = {'RewriteTensorize', 'RewriteUnboundBlock'}
    pps = [pp for pp in postprocs if type(pp).__name__ not in skip]
    pp_names = [type(p).__name__ for p in pps]
    print(f'Postproc: {" -> ".join(pp_names)}')
    print(f'\nSampling until 1st valid or {max_attempts} attempts...\n')

    # Step 2: Sample until first pass or max_attempts
    n_pass = 0
    fail_reasons = Counter()
    first_pass_attempt = None
    passed_tile_params = []

    t0 = time.time()
    random.seed(42)

    attempt = 0
    while n_pass == 0 and attempt < max_attempts:
        attempt += 1
        ds_idx = random.randint(0, len(trace_insts) - 1)
        trace = Trace(trace_insts[ds_idx], {})

        ok = False
        bad = None
        tile_info = None
        try:
            sch = Schedule(Matmul128, debug_mask='all')
            trace.apply_to_schedule(sch, remove_postproc=True)
            sch.enter_postproc()

            # Extract tile decisions before postproc modifies them
            decisions = sch.trace.decisions
            tile_info = {}
            for key, val in list(decisions.items()):
                if 'sample_perfect_tile' in str(key).lower():
                    try:
                        tile_info[str(key)[:80]] = tuple(
                            int(v.value) if hasattr(v, 'value') else int(v) for v in list(val))
                    except: pass

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
            n_pass += 1
            first_pass_attempt = attempt
            passed_tile_params.append(tile_info)
            print(f'\n  *** FIRST VALID at attempt {attempt}! ***')
            if tile_info:
                for k, v in tile_info.items():
                    print(f'      {k}: {v}')
        else:
            fail_reasons[bad or '?'] += 1

        if attempt % 200 == 0:
            elapsed = time.time() - t0
            print(f'  [{attempt:4d}] pass={n_pass:3d} | {elapsed:.0f}s')

    elapsed = time.time() - t0

    print(f'\n{"="*60}')
    print(f'RESULTS: {attempt} attempts, {elapsed:.0f}s')
    if n_pass > 0:
        print(f'  First valid at attempt: {first_pass_attempt}')
        print(f'  Effective rate: {n_pass/attempt*100:.2f}%')
    else:
        print(f'  No valid candidates in {max_attempts} attempts')
    print(f'  Fail reasons:')
    for k, v in fail_reasons.most_common():
        print(f'    {k}: {v}')

    return n_pass, attempt, dict(fail_reasons), first_pass_attempt, passed_tile_params


if __name__ == '__main__':
    p, n, reasons, first, tile_params = run(1000)

    # Save
    od = 'F:/tvm/experiments/data'
    os.makedirs(od, exist_ok=True)
    with open(f'{od}/exp01_random_sample.json', 'w') as f:
        json.dump({'attempts': n, 'pass': p, 'first_pass_at': first,
                   'reasons': reasons, 'tile_params': tile_params}, f, indent=2, default=str)
    print(f'Saved: {od}/exp01_random_sample.json')
