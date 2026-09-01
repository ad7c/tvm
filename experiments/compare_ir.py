"""Compare IR: valid tune_tir record vs random failed attempt."""
import sys,os,json,re
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

pps=[pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]

# === Load valid record from tune_tir database ===
with open('F:/tvm/experiments/data/tune_demo2/database_tuning_record.json','r') as f:
    line=f.readline()
rec=json.loads(line)
full_data=rec[1]
trace_valid=[full_data[0], full_data[1]]
# Run postproc on it too, check buffer size
sch1=Schedule(M,debug_mask='all')
TraceApplyJSONToSchedule(trace_valid,sch1)
sch1.enter_postproc()
ok1=True
for pp in pps:
    if not pp.apply(sch1):ok1=False;break
ir1=sch1.mod.script()

# === Failed attempt (random) ===
from tvm.s_tir.schedule import Schedule,Trace
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply
import random
sg=PostOrderApply();ctx=TuneContext(mod=M,target=tgt,space_generator=sg,task_name='cmp')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
insts=[d.trace.insts for d in ds]
random.seed(42)
t=Trace(insts[random.randint(0,2)],{})
sch2=Schedule(M,debug_mask='all');t.apply_to_schedule(sch2,remove_postproc=False)
sch2.enter_postproc()
ok2=True
for pp in pps:
    if not pp.apply(sch2):ok2=False;break
ir2=sch2.mod.script()

# === Compare buffer allocations ===
print(f'VALID RECORD (postproc {"PASS" if ok1 else "FAIL"}):')
for line in ir1.split('\n'):
    if 'alloc_buffer' in line:
        print(f'  {line.strip()[:150]}')
    if 'thread_binding' in line:
        print(f'  {line.strip()[:150]}')

print(f'\nFAILED RANDOM (postproc {"PASS" if ok2 else "FAIL"}):')
for line in ir2.split('\n'):
    if 'alloc_buffer' in line:
        print(f'  {line.strip()[:150]}')
    if 'thread_binding' in line:
        print(f'  {line.strip()[:150]}')

# Count shared memory
def count_shmem(ir):
    total=0
    for m in re.finditer(r'alloc_buffer\(\((\d+),\s*(\d+)\).*shared',ir):
        total+=int(m.group(1))*int(m.group(2))*4
    return total
print(f'\nVALID shared memory: {count_shmem(ir1)} bytes')
print(f'FAILED shared memory: {count_shmem(ir2)} bytes')
