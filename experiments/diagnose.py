"""Diagnose: print every attempt to find why baseline finds 0."""
import sys,os,random
os.environ['USERPROFILE']='C:\\tvm_home';os.environ['HOME']='C:\\tvm_home'
sys.path.insert(0,'F:/tvm/python');os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
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
class Matmul128:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(128,128),'float32');B=T.match_buffer(b,(128,128),'float32');C=T.match_buffer(c,(128,128),'float32')
        for i,j,k in T.grid(128,128,128):
            with T.sblock('matmul'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

TARGET=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})

sg=PostOrderApply()
ctx=TuneContext(mod=Matmul128,target=TARGET,space_generator=sg,task_name='diag')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(Matmul128)
insts=[d.trace.insts for d in ds]
print(f'DS: {len(insts)}')

postprocs=Postproc.create('cuda')
skip={'RewriteTensorize','RewriteUnboundBlock'}
pps=[pp for pp in postprocs if type(pp).__name__ not in skip]
pp_names=[type(p).__name__ for p in pps]
print(f'Postproc: {pp_names}')

random.seed(42)
import time
for attempt in range(1,21):
    t=Trace(insts[random.randint(0,2)],{})
    t0=time.time()
    try:
        sch=Schedule(Matmul128,debug_mask='all')
        t.apply_to_schedule(sch,remove_postproc=False)  # False = keep postproc, like exp03
        # Extract tile info
        dm=sch.trace.decisions
        factors=[]
        for key,val in list(dm.items()):
            if 'sample_perfect_tile' in str(key).lower():
                try:factors.append(tuple(int(v.value)if hasattr(v,'value')else int(v)for v in list(val)))
                except:pass
        sch.enter_postproc()
        ok=True;bad=None
        for pp in pps:
            pp_name=type(pp).__name__
            if not pp.apply(sch):
                ok=False;bad=pp_name;break
        elapsed=time.time()-t0
        print(f'  [{attempt:3d}] factors={factors} {"PASS" if ok else bad} ({elapsed:.1f}s)',flush=True)
    except Exception as e:
        print(f'  [{attempt:3d}] EXC: {type(e).__name__}: {e}',flush=True)
