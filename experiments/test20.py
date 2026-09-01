"""Run until first valid found (max 600)."""
import sys,os,time,random
from collections import Counter
os.environ['USERPROFILE']='C:\\tvm_home';os.environ['HOME']='C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
sys.path.insert(0,'F:/tvm/python')
from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build
register_global_func('default')(default_build)
import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.schedule import Schedule,Trace
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply
from tvm.s_tir.meta_schedule.postproc import Postproc

@tvm.script.ir_module
class M:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(128,128),'float32');B=T.match_buffer(b,(128,128),'float32');C=T.match_buffer(c,(128,128),'float32')
        for i,j,k in T.grid(128,128,128):
            with T.sblock('mm'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

tgt=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})
sg=PostOrderApply();ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='hunt')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
insts=[d.trace.insts for d in ds]
pps=[pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]
print(f'DS:{len(insts)}, PP:{len(pps)}, hunting for 1st valid...')

random.seed(42);t0=time.time();MAX=600
for a in range(1,MAX+1):
    t=Trace(insts[random.randint(0,2)],{})
    try:
        sch=Schedule(M,debug_mask='all');t.apply_to_schedule(sch,remove_postproc=False)
        sch.enter_postproc();ok=True
        for pp in pps:
            if not pp.apply(sch):ok=False;break
        if ok:
            elapsed=time.time()-t0
            dm=sch.trace.decisions
            factors=[]
            for k,v in list(dm.items()):
                if 'sample_perfect_tile' in str(k).lower():
                    try:factors.append(tuple(int(vv.value)if hasattr(vv,'value')else int(vv)for vv in list(v)))
                    except:pass
            print(f'\n  FOUND at attempt {a}! ({elapsed:.0f}s)')
            print(f'  Tile factors: {factors}')
            break
    except Exception as e:
        if a<=3:print(f'  [{a}] EXC {type(e).__name__}')
    if a%100==0:print(f'  [{a}] {time.time()-t0:.0f}s',flush=True)

print(f'{a} attempts, {"PASS" if ok else "no pass"}, {time.time()-t0:.0f}s')
