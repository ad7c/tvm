"""
Exp: Baseline only — verify Meta Schedule finds valid candidates.
MatMul 64^3, 5000 attempts, find 3 valid.
"""
import sys,os,time,random,json
from collections import Counter

os.environ['USERPROFILE']='C:\\tvm_home'
os.environ['HOME']='C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
sys.path.insert(0,'F:/tvm/python')

from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build,default_export
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
class MatMul128:
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

MOD = MatMul128

# Setup design spaces
sg = PostOrderApply()
ctx = TuneContext(mod=MOD, target=TARGET, space_generator=sg, task_name='baseline128')
ctx.space_generator._initialize_with_tune_context(ctx)
ds_list = ctx.space_generator.generate_design_space(MOD)
insts = [d.trace.insts for d in ds_list]
print(f'Design spaces: {len(insts)}')

# Postproc
pps = [pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]
ppn = [type(p).__name__ for p in pps]
print(f'Postproc: {ppn}\n')

# Baseline: no prefilter, SampleInitPopulation logic
MAX = 5000; NEED = 3
random.seed(42)
valid_at = []; stats = Counter()
t0 = time.time()

for attempt in range(1, MAX + 1):
    t = Trace(insts[random.randint(0, 2)], {})
    try:
        sch = Schedule(MOD, debug_mask='all')
        t.apply_to_schedule(sch, remove_postproc=False)
        sch.enter_postproc()
        ok = True
        for pp in pps:
            if not pp.apply(sch):
                ok = False; stats[type(pp).__name__] += 1; break
        if ok:
            valid_at.append(attempt)
            stats['passed'] += 1
            elapsed = time.time() - t0
            print(f'  [{attempt:5d}] PASS #{len(valid_at)}! ({elapsed:.0f}s)', flush=True)
            if len(valid_at) >= NEED: break
    except Exception as e:
        stats[f'exc_{type(e).__name__}'] += 1

    if attempt % 500 == 0 or attempt <= 5:
        elapsed = time.time() - t0
        print(f'  [{attempt:5d}] valid={len(valid_at)} {elapsed:.0f}s', flush=True)

elapsed = time.time() - t0
print(f'\n{"="*50}')
print(f'BASELINE: {attempt} attempts, {len(valid_at)} valid, {elapsed:.0f}s')
print(f'Stats: {dict(stats.most_common())}')
if valid_at:
    print(f'Valid at: {valid_at}')

# Save
od = 'F:/tvm/experiments/data'
os.makedirs(od, exist_ok=True)
with open(f'{od}/baseline_64.json', 'w') as f:
    json.dump({'attempts': attempt, 'valid': len(valid_at), 'valid_at': valid_at,
               'time_s': elapsed, 'stats': dict(stats)}, f, indent=2)
print(f'Saved: {od}/baseline_64.json')
