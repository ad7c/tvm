"""Test basic TVM CUDA build without Meta Schedule."""
import sys, os, numpy as np, time
os.environ['USERPROFILE'] = 'C:\\tvm_home'
os.environ['HOME'] = 'C:\\tvm_home'
sys.path.insert(0, 'F:/tvm/python')
os.environ['TVM_LIBRARY_PATH'] = 'F:/tvm/build/lib'

import tvm
from tvm.script import tirx as T
from tvm.target import Target


@tvm.script.ir_module
class Matmul:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({'global_symbol': 'main'})
        A = T.match_buffer(a, (256, 256), 'float32')
        B = T.match_buffer(b, (256, 256), 'float32')
        C = T.match_buffer(c, (256, 256), 'float32')
        for i, j, k in T.grid(256, 256, 256):
            with T.sblock('matmul'):
                vi, vj, vk = T.axis.remap('SSR', [i, j, k])
                with T.init(): C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


print('=== Test 1: Raw build (no schedule) ===')
tgt = Target({'kind': 'cuda', 'arch': 'sm_86'})
try:
    rt = tvm.build(Matmul, target=tgt)
    print(f'Build OK, entry: {rt.entry_name}')
except Exception as e:
    print(f'Build FAIL: {type(e).__name__}: {str(e)[:200]}')

print('\n=== Test 2: With MultiLevelTiling schedule (no postproc) ===')
from tvm.s_tir.schedule import Schedule
from tvm.s_tir.meta_schedule.schedule_rule.multi_level_tiling import MultiLevelTiling
from tvm.s_tir.meta_schedule import TuneContext

TARGET = Target({
    'kind': 'cuda', 'arch': 'sm_86',
    'max_threads_per_block': 1024,
    'max_shared_memory_per_block': 49152,
    'thread_warp_size': 32,
})

mlt = MultiLevelTiling(
    structure='SSSRRSRS',
    tile_binds=['blockIdx.x', 'vthread.x', 'threadIdx.x'],
    max_innermost_factor=64,
)

ctx = TuneContext(mod=Matmul, target=TARGET, space_generator='post-order-apply', task_name='test')
mlt._initialize_with_tune_context(ctx)

# Try multiple times
for trial in range(5):
    try:
        sch = Schedule(Matmul, debug_mask='all')
        block = sch.get_sblock('matmul')
        states = mlt.apply(sch, block)
        if states:
            print(f'  Trial {trial}: MLT applied, {len(states)} states')
            # Try to build
            try:
                rt2 = tvm.build(sch.mod, target=TARGET)
                print(f'    Build OK: {rt2.entry_name}')
                # Benchmark
                dev = tvm.cuda(0)
                a = np.random.randn(256, 256).astype('float32')
                b = np.random.randn(256, 256).astype('float32')
                c = np.zeros((256, 256), 'float32')
                a_g = tvm.runtime.ndarray.array(a, dev)
                b_g = tvm.runtime.ndarray.array(b, dev)
                c_g = tvm.runtime.ndarray.array(c, dev)
                te = rt2.time_evaluator(rt2.entry_name, dev, number=10, repeat=3)
                result = te(a_g, b_g, c_g)
                print(f'    Latency: {result.mean*1000:.2f} ms')
                break
            except Exception as e:
                print(f'    Build FAIL: {type(e).__name__}: {str(e)[:150]}')
    except Exception as e:
        print(f'  Trial {trial}: MLT FAIL: {type(e).__name__}: {str(e)[:150]}')
