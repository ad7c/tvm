"""
Root cause: design space trace places shared memory buffer at outermost loop
→ alloc(128×128) = 64KB each → 100% VerifyGPUCode fail.

Fix: use Trace(insts, {}) with remove_postproc=False (same as Exp03 which worked).
Test 500 random samples to confirm.
"""
import sys,os,time,random,re
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
        A=T.match_buffer(a,(128,128),'float32');B=T.match_buffer(b,(128,128),'float32');C=T.match_buffer(c,(128,128),'float32')
        for i,j,k in T.grid(128,128,128):
            with T.sblock('mm'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

tgt=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})
sg=PostOrderApply();ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='fix')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
insts=[d.trace.insts for d in ds]

pps=[pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]
ppn=[type(p).__name__ for p in pps]
print(f'Postproc: {ppn}')

random.seed(42)
t0=time.time()
n_pass=0;stats=Counter()
for attempt in range(1,501):
    t=Trace(insts[random.randint(0,2)],{})
    try:
        sch=Schedule(M,debug_mask='all')
        t.apply_to_schedule(sch,remove_postproc=False)

        # Extract tile params BEFORE postproc
        dm=sch.trace.decisions;sp_factors=[]
        for key,val in list(dm.items()):
            if 'sample_perfect_tile' in str(key).lower():
                try:sp_factors.append(tuple(int(v.value)if hasattr(v,'value')else int(v)for v in list(val)))
                except:pass

        sch.enter_postproc();ok=True;bad=None
        for pp in pps:
            if not pp.apply(sch):ok=False;bad=type(pp).__name__;break
        if ok:
            n_pass+=1
            print(f'  [{attempt:4d}] PASS! factors={sp_factors}')
            break
        else:
            stats[bad]+=1
            if attempt<=3:
                ir=sch.mod.script();allocs=[]
                for l in ir.split('\n'):
                    if 'alloc_buffer' in l:
                        m=re.search(r'alloc_buffer\(\((\d+),\s*(\d+)\).*(shared|local)',l)
                        if m:allocs.append(f'{m.group(1)}x{m.group(2)} {m.group(3)}')
                print(f'  [{attempt:4d}] factors={sp_factors} fail={bad} allocs={allocs}')
    except Exception as e:
        stats[f'exc_{type(e).__name__}']+=1
    if attempt%100==0:
        print(f'  [{attempt:4d}] {time.time()-t0:.0f}s pass={n_pass} fails={dict(stats.most_common(3))}')

if n_pass==0:
    print(f'\nFAILED all {attempt} attempts')
