"""Replay a tune_tir record to verify it passes postproc."""
import sys,os,json
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

# Read first record
with open('F:/tvm/experiments/data/tune_demo2/database_tuning_record.json','r') as f:
    line=f.readline()
rec=json.loads(line)
full_data=rec[1]

# Debug structure
print(f'rec[0] (task_id): {rec[0]}')
print(f'rec[1] type: {type(full_data)}, len: {len(full_data)}')
for i,item in enumerate(full_data):
    print(f'  full_data[{i}]: type={type(item).__name__}, len={len(item) if hasattr(item,"__len__") else "N/A"}')
    if i<2 and hasattr(item,'__len__') and len(item)<10:
        print(f'    value: {item}')

# Try different formats
# Format might be: [instructions, decisions, run_secs, target, args]
# So trace = [full_data[0], full_data[1]]
trace_data = [full_data[0], full_data[1]]
print(f'\nTrace[0] (insts) len: {len(trace_data[0])}')
print(f'Trace[1] (decs) len: {len(trace_data[1])}')
if len(trace_data[1])>0:
    print(f'  First dec: {trace_data[1][0]}')

# Replay
sch=Schedule(M,debug_mask='all')
TraceApplyJSONToSchedule(trace_data,sch)
print('Trace replayed')

# Postproc
pps=[pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]
sch.enter_postproc()
ok=True
for pp in pps:
    pp_name=type(pp).__name__
    if not pp.apply(sch):
        print(f'  {pp_name}: REJECT')
        ok=False;break
    print(f'  {pp_name}: PASS')

if ok:
    print('\nALL PASSED! Building...')
    rt=tvm.build(sch.mod,target=tgt)
    import numpy as np,time
    dev=tvm.cuda(0)
    a=np.random.randn(128,128).astype('float32')
    b=np.random.randn(128,128).astype('float32')
    c=np.zeros((128,128),'float32')
    a_g=tvm.runtime.ndarray.array(a,dev)
    b_g=tvm.runtime.ndarray.array(b,dev)
    c_g=tvm.runtime.ndarray.array(c,dev)
    te=rt.time_evaluator(rt.entry_name,dev,number=10,repeat=3)
    r=te(a_g,b_g,c_g)
    err=np.max(np.abs(c_g.numpy()-a@b))
    print(f'Latency: {r.mean*1000:.3f}ms, GFLOPS: {2*128**3/(r.mean*1e9):.1f}, err={err:.2e}')
else:
    print('POSTPROC FAILED')
