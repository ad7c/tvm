"""Dump lowered IR for first failing candidate."""
import sys,os,re
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
sg=PostOrderApply();ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='t2')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
base=ds[0].trace.as_json();insts,decs=base[0],base[1]
sp=[i for i,inst in enumerate(insts) if inst[0]=='SamplePerfectTile'and inst[2][0]==5]
rd=[i for i,inst in enumerate(insts) if inst[0]=='SamplePerfectTile'and inst[2][0]==3]

# Test (4,4,8) vk=4
bi,ti,bj,tj,bk,vk,rk=4,8,4,8,8,4,4
si=[bi,1,ti,1,1];sj=[bj,1,tj,1,1];sk=[bk,vk,rk]
nd=[]
for d in decs:
    if d[0]==sp[0]:nd.append([d[0],si])
    elif d[0]==sp[1]:nd.append([d[0],sj])
    elif d[0]==rd[0]:nd.append([d[0],sk])
    else:nd.append(d)

pps=[pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]
pp_names=[type(p).__name__ for p in pps]

sch=Schedule(M,debug_mask='all')
TraceApplyJSONToSchedule([insts,nd],sch)
print("=== IR BEFORE POSTPROC ===")
print(sch.mod.script()[:2000])

sch.enter_postproc()
for pp in pps:
    pp_name=type(pp).__name__
    ok=pp.apply(sch)
    print(f"\n=== AFTER {pp_name} (ok={ok}) ===")
    if not ok:
        ir=sch.mod.script()
        # Find resource usage
        threads_x=threads_y=threads_z=1
        shmem_total=0;local_total=0
        for line in ir.split('\n'):
            m=re.search(r'thread_binding\((\d+).*threadIdx\.(\w)',line)
            if m:
                v=int(m.group(1));axis=m.group(2)
                if axis=='x':threads_x=max(threads_x,v)
                elif axis=='y':threads_y=max(threads_y,v)
                elif axis=='z':threads_z=max(threads_z,v)
            if 'alloc_buffer' in line:
                sz=1
                for n in re.findall(r'\((\d+)',line):sz*=int(n)
                if 'shared' in line:shmem_total+=sz*4
                if 'local' in line:local_total+=sz*4
                print(f'  ALLOC: {line.strip()[:150]}')
            if 'vectorized' in line:
                m2=re.search(r'vectorized\((\d+)\)',line)
                if m2:print(f'  VEC: {m2.group(1)}')
            if 'thread_binding' in line:
                print(f'  BIND: {line.strip()[:150]}')
        total_threads=threads_x*threads_y*threads_z
        print(f'\n  Threads: x={threads_x} y={threads_y} z={threads_z} total={total_threads}')
        print(f'  Shared mem: {shmem_total} (limit 49152)')
        print(f'  Local mem: {local_total}')
        break
