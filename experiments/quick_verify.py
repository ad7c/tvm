"""10-second check: can design space trace pass postproc?"""
import sys,os;os.environ['USERPROFILE']='C:\\tvm_home';os.environ['HOME']='C:\\tvm_home'
sys.path.insert(0,'F:/tvm/python');os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build
register_global_func('default')(default_build)
import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.schedule import Schedule
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply
from tvm.s_tir.meta_schedule.postproc import Postproc

@tvm.script.ir_module
class M:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({'global_symbol': 'main'})
        A = T.match_buffer(a, (128, 128), 'float32')
        B = T.match_buffer(b, (128, 128), 'float32')
        C = T.match_buffer(c, (128, 128), 'float32')
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock('matmul'):
                vi, vj, vk = T.axis.remap('SSR', [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

tgt=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})
sg=PostOrderApply();ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='qv')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
print(f'DS: {len(ds)}')

postprocs=Postproc.create('cuda');skip={'RewriteTensorize','RewriteUnboundBlock'}
pps=[pp for pp in postprocs if type(pp).__name__ not in skip]
pp_names=[type(p).__name__ for p in pps]

for ds_idx in range(len(ds)):
    for remove_pp in [False, True]:
        try:
            sch=Schedule(M,debug_mask='all')
            ds[ds_idx].trace.apply_to_schedule(sch,remove_postproc=remove_pp)
            sch.enter_postproc();ok=True;bad=None
            for pp in pps:
                if not pp.apply(sch):ok=False;bad=type(pp).__name__;break
            print(f'  DS[{ds_idx}] remove_postproc={remove_pp}: {"PASS" if ok else f"FAIL({bad})"}')
        except Exception as e:
            print(f'  DS[{ds_idx}] remove_postproc={remove_pp}: EXC {type(e).__name__}')
