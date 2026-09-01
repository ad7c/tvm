"""Debug: inspect WHY VerifyGPUCode rejects pre-filter-passed candidates."""
import sys, os
os.environ['USERPROFILE']='C:\\tvm_home'; os.environ['HOME']='C:\\tvm_home'
sys.path.insert(0,'F:/tvm/python'); os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
sys.path.insert(0,'F:/tvm/experiments')
from pruning import check_upward, extract_tile_params
from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build, default_export
register_global_func('default')(default_build); register_global_func('default_export')(default_export)
import tvm,re,random
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
sg=PostOrderApply(); ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='dbg')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
insts=[d.trace.insts for d in ds]
postprocs=Postproc.create('cuda'); skips={'RewriteTensorize','RewriteUnboundBlock'}
pps=[pp for pp in postprocs if type(pp).__name__ not in skips]

# Find some pre-filter-passed candidates that fail VerifyGPUCode
n_found=0
for attempt in range(500):
    if n_found>=3: break
    t=Trace(insts[random.randint(0,2)],{})
    try:
        sch=Schedule(M,debug_mask='all'); t.apply_to_schedule(sch,remove_postproc=True)
        bi,bj,bk,thr=extract_tile_params(sch.trace)
        ok,_=check_upward(bi,bj,bk,thr)
        if not ok: continue
        # Pre-filter passed. Now test postproc
        sch.enter_postproc()
        passed=True; bad=None
        for pp in pps:
            if not pp.apply(sch): passed=False; bad=type(pp).__name__; break
        if not passed and bad=='VerifyGPUCode':
            n_found+=1
            print(f'\n=== Candidate {n_found}: bi={bi}, bj={bj}, bk={bk}, threads={thr} ===')
            ir=sch.mod.script()
            # Find thread bindings and memory allocations
            threads_x=threads_y=threads_z=1
            shmem_total=0
            for line in ir.split('\n'):
                m=re.search(r'T\.thread_binding\((\d+).*threadIdx\.(\w)',line)
                if m:
                    val=int(m.group(1)); axis=m.group(2)
                    if axis=='x': threads_x=max(threads_x,val)
                    elif axis=='y': threads_y=max(threads_y,val)
                    elif axis=='z': threads_z=max(threads_z,val)
                if 'alloc_buffer' in line and 'shared' in line:
                    m2=re.search(r'alloc_buffer\(\((\d+),\s*(\d+)\)',line)
                    if m2: shmem_total+=int(m2.group(1))*int(m2.group(2))*4
                    print(f'  SHMEM: {line.strip()[:120]}')
                if 'alloc_buffer' in line and 'local' in line:
                    print(f'  LOCAL: {line.strip()[:120]}')
                if 'vectorized' in line:
                    m3=re.search(r'vectorized\((\d+)\)',line)
                    if m3: print(f'  VEC: {line.strip()[:120]}')
            print(f'  Threads: x={threads_x} y={threads_y} z={threads_z} total={threads_x*threads_y*threads_z}')
            print(f'  Shared memory: ~{shmem_total} bytes (limit 49152)')
    except: pass