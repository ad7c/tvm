"""Brute-force: enumerate ALL block tile combos for MatMul 64^3, test VerifyGPUCode."""
import sys,os,json,math
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
        A=T.match_buffer(a,(64,64),'float32');B=T.match_buffer(b,(64,64),'float32');C=T.match_buffer(c,(64,64),'float32')
        for i,j,k in T.grid(64,64,64):
            with T.sblock('mm'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

tgt=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})
sg=PostOrderApply();ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='b64')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
base=ds[0].trace.as_json();insts,decs=base[0],base[1]
sp=[i for i,inst in enumerate(insts) if inst[0]=='SamplePerfectTile'and inst[2][0]==5]
rd=[i for i,inst in enumerate(insts) if inst[0]=='SamplePerfectTile'and inst[2][0]==3]
print(f'SP:{sp} RD:{rd}')

pps=[pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]

divs=[1,2,4,8,16,32,64]
total=0;passed=0;failed={}
results=[]

for bi in divs:
    si=[bi,1,64//bi,1,1]
    for bj in divs:
        tj=64//bj;threads=(64//bi)*(64//bj)
        if threads<32 or threads>1024:continue
        sj=[bj,1,tj,1,1]
        for bk in divs:
            for vk_val in divs:
                rk_val=64//(bk*vk_val)
                if bk*vk_val*rk_val!=64:continue
                total+=1
                sk=[bk,vk_val,rk_val]
                nd=[]
                for d in decs:
                    if d[0]==sp[0]:nd.append([d[0],list(si)])
                    elif d[0]==sp[1]:nd.append([d[0],list(sj)])
                    elif d[0]==rd[0]:nd.append([d[0],list(sk)])
                    else:nd.append(d)
                try:
                    sch=Schedule(M,debug_mask='all')
                    TraceApplyJSONToSchedule([insts,nd],sch)
                    sch.enter_postproc();ok=True;bad=None
                    for pp in pps:
                        if not pp.apply(sch):ok=False;bad=type(pp).__name__;break
                    if ok:
                        passed+=1
                        results.append((bi,bj,bk,vk_val,threads))
                        print(f'  PASS: ({bi},{bj},{bk}) vk={vk_val} thr={threads}')
                    else:
                        failed[bad]=failed.get(bad,0)+1
                except:failed['exc']=failed.get('exc',0)+1

print(f'\nTotal tested: {total}')
print(f'Passed: {passed} ({passed/total*100:.1f}%)')
print(f'Failed: {failed}')

if results:
    print(f'\nPassing configs:')
    for r in results:print(f'  bi={r[0]} bj={r[1]} bk={r[2]} vk={r[3]} thr={r[4]}')
