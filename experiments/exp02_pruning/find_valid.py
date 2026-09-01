"""Brute-force: which tile params produce legal shared memory in the lowered IR?"""
import sys,os,random,re,time
os.environ['USERPROFILE']='C:\\tvm_home'; os.environ['HOME']='C:\\tvm_home'
sys.path.insert(0,'F:/tvm/python'); os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
sys.path.insert(0,'F:/tvm/experiments/exp02_pruning')
try:
    from pruning import extract_tile_params, estimate_shmem, check_upward
except Exception as e:
    print(f'pruning import failed: {e}')
    raise
from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build, default_export
register_global_func('default')(default_build); register_global_func('default_export')(default_export)
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
            with T.sblock('m'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

tgt=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})
sg=PostOrderApply();ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='fv')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
insts=[d.trace.insts for d in ds]
postprocs=Postproc.create('cuda')
skips={'RewriteTensorize','RewriteUnboundBlock'}
pps=[pp for pp in postprocs if type(pp).__name__ not in skips]

# Sample until we find 5 valid
valid_count=0; total=0
t0=time.time()
while valid_count<5 and total<5000:
    total+=1
    ds_idx=random.randint(0,2)
    t=Trace(insts[ds_idx],{})
    try:
        sch=Schedule(M,debug_mask='all'); t.apply_to_schedule(sch,remove_postproc=True)
        bi,bj,bk,thr=extract_tile_params(sch.trace)
        sch.enter_postproc()
        ok=True
        for pp in pps:
            if not pp.apply(sch): ok=False; break
        if ok:
            valid_count+=1
            # Inspect the successful schedule
            ir=sch.mod.script()
            shmem_lines=[]
            for line in ir.split('\n'):
                if 'shared' in line and 'alloc' in line:
                    shmem_lines.append(line.strip()[:150])
            print(f'\nVALID #{valid_count} at attempt {total}:')
            print(f'  bi={bi}, bj={bj}, bk={bk}, threads={thr}')
            print(f'  Our estimate_shmem: {estimate_shmem(bi,bj,bk)} bytes')
            print(f'  Actual shmem allocs:')
            for sl in shmem_lines: print(f'    {sl}')
    except: pass
    if total%500==0: print(f'  [{total}] valid={valid_count} ({time.time()-t0:.0f}s)')

print(f'\nDone: {valid_count} valid in {total} attempts ({time.time()-t0:.0f}s)')
