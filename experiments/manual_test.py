"""Manually construct a candidate that SHOULD pass VerifyGPUCode."""
import sys,os
os.environ['USERPROFILE']='C:\\tvm_home';os.environ['HOME']='C:\\tvm_home'
sys.path.insert(0,'F:/tvm/python');os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build
register_global_func('default')(default_build)
import tvm,json,re
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
sg=PostOrderApply();ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='man')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)

# Get trace JSON, find SP/RD indices, modify decisions
base=ds[0].trace.as_json()
insts=base[0];decs=base[1]
sp_idx=[i for i,inst in enumerate(insts) if inst[0]=='SamplePerfectTile' and inst[2][0]==5]
rd_idx=[i for i,inst in enumerate(insts) if inst[0]=='SamplePerfectTile' and inst[2][0]==3]
print(f'SP: {sp_idx}, RD: {rd_idx}')

# Test a grid of manually-chosen params
tests=[]
# i: (bi, vi, ti, si1, si2), j: same, k: (bk, vk, rk)
for bi in [4,8,16,32]:
    ti=128//bi
    for bj in [4,8,16,32]:
        tj=128//bj;threads=ti*tj
        if threads<32 or threads>1024:continue
        for bk in [8,16,32,64]:
            if 128%bk!=0:continue
            for vk in [2,4,8,16,32]:
                rk=128//(bk*vk)
                if bk*vk*rk!=128:continue
                if vk==1:continue
                shmem=(bi*bk+bk*bj)*8+bi*bj*4
                if shmem>49152:continue
                tests.append((bi,ti,bj,tj,bk,vk,rk,threads,shmem))

print(f'Testing {len(tests)} manual candidates...\n')

postprocs=Postproc.create('cuda');skips={'RewriteTensorize','RewriteUnboundBlock'}
pps=[pp for pp in postprocs if type(pp).__name__ not in skips]
pp_names=[type(p).__name__ for p in pps]

n_pass=0;n_fail=0;fail_reasons={}
for idx,(bi,ti,bj,tj,bk,vk,rk,threads,shmem) in enumerate(tests):
    if n_pass>=3 and n_fail>=3:break
    # Build modified decisions: SP i, SP j, RD k
    si=[bi,1,ti,1,1];sj=[bj,1,tj,1,1];sk=[bk,vk,rk]
    new_decs=[]
    for d in decs:
        if d[0]==sp_idx[0]:new_decs.append([d[0],si])
        elif d[0]==sp_idx[1]:new_decs.append([d[0],sj])
        elif d[0]==rd_idx[0]:new_decs.append([d[0],sk])
        else:new_decs.append(d)
    mod_data=[insts,new_decs]
    try:
        sch=Schedule(M,debug_mask='all')
        TraceApplyJSONToSchedule(mod_data,sch)
        sch.enter_postproc();ok=True;bad=None
        for pp in pps:
            if not pp.apply(sch):ok=False;bad=type(pp).__name__;break
        if ok:
            n_pass+=1
            ir=sch.mod.script()
            shmem_actual=0
            for m in re.finditer(r'alloc_buffer\(\((\d+),\s*(\d+)\).*shared',ir):
                shmem_actual+=int(m.group(1))*int(m.group(2))*4
            print(f'  PASS #{n_pass}: bi={bi} bj={bj} bk={bk} vk={vk} | thr={threads} shmem={shmem} actual_shmem={shmem_actual}')
        else:
            n_fail+=1
            fail_reasons[bad]=fail_reasons.get(bad,0)+1
            if n_fail<=5:
                print(f'  FAIL #{n_fail}: bi={bi} bj={bj} bk={bk} vk={vk} | thr={threads} shmem={shmem} -> {bad}')
    except Exception as e:
        n_fail+=1
        fail_reasons[f'exc_{type(e).__name__}']=fail_reasons.get(f'exc_{type(e).__name__}',0)+1

print(f'\nDone: {n_pass} passed, {n_fail} failed')
print(f'Fail reasons: {fail_reasons}')
