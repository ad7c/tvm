"""Quick 50-attempt test to verify baseline works now."""
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
from tvm.s_tir.schedule import Schedule
from tvm.s_tir.schedule.trace import Trace
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply
from tvm.s_tir.meta_schedule.postproc import Postproc

@tvm.script.ir_module
class M:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(64,64),'float32');B=T.match_buffer(b,(64,64),'float32');C=T.match_buffer(c,(64,64),'float32')
        for i,j,k in T.grid(64,64,64):
            with T.sblock('mm'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

tgt=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})
sg=PostOrderApply();ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='qt')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
insts=[d.trace.insts for d in ds]

pps=[pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]
print(f'Postproc: {[type(p).__name__ for p in pps]}')

random.seed(42);t0=time.time();n_pass=0;stats=Counter()
for attempt in range(1,51):
    t=Trace(insts[random.randint(0,2)],{})
    try:
        sch=Schedule(M,debug_mask='all');t.apply_to_schedule(sch,remove_postproc=False)
        sch.enter_postproc();ok=True;bad=None
        for pp in pps:
            if not pp.apply(sch):ok=False;bad=type(pp).__name__;break
        if ok:
            n_pass+=1
            dm=sch.trace.decisions;factors=[]
            for k,v in list(dm.items()):
                if 'sample_perfect_tile' in str(k).lower():
                    try:factors.append(tuple(int(vv.value)if hasattr(vv,'value')else int(vv)for vv in list(v)))
                    except:pass
            print(f'  [{attempt}] PASS! factors={factors}')
        else:stats[bad]+=1
    except Exception as e:stats[f'exc_{type(e).__name__}']+=1

print(f'\n{attempt} attempts, {n_pass} passed ({n_pass/attempt*100:.1f}%) in {time.time()-t0:.0f}s')
print(f'Fails: {dict(stats.most_common())}')
