"""Minimal test: manually chosen tile params → VerifyGPUCode?"""
import sys,os
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
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply
from tvm.s_tir.meta_schedule.postproc import Postproc
from tvm.s_tir.schedule._ffi_api import TraceApplyJSONToSchedule

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
sg=PostOrderApply();ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='t')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
base=ds[0].trace.as_json();insts,decs=base[0],base[1]
sp=[i for i,inst in enumerate(insts) if inst[0]=='SamplePerfectTile'and inst[2][0]==5]
rd=[i for i,inst in enumerate(insts) if inst[0]=='SamplePerfectTile'and inst[2][0]==3]
print(f'SP:{sp} RD:{rd}')

pps=[pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]
ppn=[type(p).__name__ for p in pps]
print(f'Postproc: {ppn}')

tests=[(4,8,4,8,8,4,4),(8,4,8,4,16,4,2),(16,2,8,4,8,8,2),(8,4,4,8,16,4,2)]
n_pass=0
for bi,ti,bj,tj,bk,vk,rk in tests:
    threads=ti*tj;shmem=(bi*bk+bk*bj)*8+bi*bj*4
    nd=[];si=[bi,1,ti,1,1];sj=[bj,1,tj,1,1];sk=[bk,vk,rk]
    for d in decs:
        if d[0]==sp[0]:nd.append([d[0],si])
        elif d[0]==sp[1]:nd.append([d[0],sj])
        elif d[0]==rd[0]:nd.append([d[0],sk])
        else:nd.append(d)
    try:
        sch=Schedule(M,debug_mask='all')
        TraceApplyJSONToSchedule([insts,nd],sch)
        sch.enter_postproc();ok=True;bad=None
        for pp in pps:
            if not pp.apply(sch):ok=False;bad=type(pp).__name__;break
        r='PASS' if ok else f'FAIL({bad})'
        print(f'  ({bi},{bj},{bk}) vk={vk} thr={threads} shmem={shmem}: {r}')
        if ok:n_pass+=1
    except Exception as e:print(f'  ({bi},{bj},{bk}) EXC: {type(e).__name__}: {e}')
print(f'\nPASSED: {n_pass}/{len(tests)}')
