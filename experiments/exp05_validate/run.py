"""
Exp 05: Collect valid candidates (full trace), build & benchmark.
"""
import sys, os, time, random, json
os.environ['USERPROFILE'] = 'C:\\tvm_home'
os.environ['HOME'] = 'C:\\tvm_home'
sys.path.insert(0, 'F:/tvm/python')
os.environ['TVM_LIBRARY_PATH'] = 'F:/tvm/build/lib'

from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build, default_export
register_global_func('default')(default_build)
register_global_func('default_export')(default_export)

import tvm, numpy as np
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


def collect_valid(n_needed=6):
    sg = PostOrderApply()
    ctx = TuneContext(mod=Matmul128, target=TARGET, space_generator=sg, task_name='exp05')
    ctx.space_generator._initialize_with_tune_context(ctx)
    ds_list = ctx.space_generator.generate_design_space(Matmul128)
    trace_insts = [ds.trace.insts for ds in ds_list]

    postprocs = Postproc.create('cuda')
    skip = {'RewriteTensorize', 'RewriteUnboundBlock'}
    pps = [pp for pp in postprocs if type(pp).__name__ not in skip]

    valid_traces = []
    random.seed(42)
    t0 = time.time()

    for attempt in range(1, 3000):
        ds_idx = random.randint(0, 2)
        trace = Trace(trace_insts[ds_idx], {})

        try:
            sch = Schedule(Matmul128, debug_mask='all')
            trace.apply_to_schedule(sch, remove_postproc=True)
            sch.enter_postproc()

            # Save trace BEFORE postproc (clean, no postproc instructions)
            clean_trace = sch.trace.as_json()

            ok = True
            for pp in pps:
                if not pp.apply(sch):
                    ok = False; break

            if ok:
                valid_traces.append(clean_trace)
                print(f'  [{attempt:5d}] VALID #{len(valid_traces)}! '
                      f'({time.time()-t0:.0f}s)', flush=True)
                if len(valid_traces) >= n_needed:
                    break
        except:
            pass

    return valid_traces


def build_and_benchmark(trace_list):
    from tvm.s_tir.schedule._ffi_api import TraceApplyJSONToSchedule
    dev = tvm.cuda(0)
    a_np = np.random.randn(128, 128).astype('float32')
    b_np = np.random.randn(128, 128).astype('float32')
    c_ref = a_np @ b_np

    results = []
    for i, trace_data in enumerate(trace_list):
        print(f'\n  Valid #{i+1}: building...', flush=True)
        try:
            sch = Schedule(Matmul128, debug_mask='all')
            TraceApplyJSONToSchedule(trace_data, sch)

            # Postproc
            postprocs = Postproc.create('cuda')
            skip = {'RewriteTensorize', 'RewriteUnboundBlock'}
            pps = [pp for pp in postprocs if type(pp).__name__ not in skip]
            sch.enter_postproc()
            for pp in pps:
                if not pp.apply(sch):
                    print(f'    Postproc REJECT: {type(pp).__name__}')
                    break
            else:
                rt = tvm.build(sch.mod, target=TARGET)
                a_g = tvm.runtime.ndarray.array(a_np, dev)
                b_g = tvm.runtime.ndarray.array(b_np, dev)
                c_g = tvm.runtime.ndarray.array(np.zeros((128,128),'float32'), dev)

                te = rt.time_evaluator(rt.entry_name, dev, number=10, repeat=5)
                result = te(a_g, b_g, c_g)
                latency_ms = result.mean * 1000
                c_out = c_g.numpy()

                # Check correctness
                max_err = np.max(np.abs(c_out - c_ref))
                ok = max_err < 1e-3
                gflops = 2 * 128**3 / (result.mean * 1e9)

                print(f'    Build OK | Latency: {latency_ms:.3f}ms | GFLOPS: {gflops:.1f} | '
                      f'Max error: {max_err:.2e} | {"CORRECT" if ok else "WRONG"}')

                results.append({'id': i+1, 'latency_ms': latency_ms, 'gflops': gflops,
                               'max_error': float(max_err), 'correct': ok})
        except Exception as e:
            print(f'    FAIL: {type(e).__name__}: {str(e)[:100]}')
            results.append({'id': i+1, 'error': str(e)[:200]})

    return results


if __name__ == '__main__':
    print('Collecting valid candidates...')
    traces = collect_valid(n_needed=6)
    print(f'\nCollected {len(traces)} valid traces')

    if traces:
        print('\nBuilding & benchmarking...')
        results = build_and_benchmark(traces)

        # Save
        od = 'F:/tvm/experiments/data'
        os.makedirs(od, exist_ok=True)
        with open(f'{od}/exp05_validate.json', 'w') as f:
            json.dump({'n_valid': len(traces), 'results': results}, f, indent=2, default=str)

        print(f'\nSaved: {od}/exp05_validate.json')
