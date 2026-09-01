"""Test if pruning DLL works — compare rejection counts."""
import sys,os,time,re
os.environ['USERPROFILE']='C:\\tvm_home';os.environ['HOME']='C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
os.environ['CUDA_HOME']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['CUDA_PATH']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['PYTHONPATH']='F:/tvm/python'
sys.path.insert(0,'F:/tvm/python')
from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build,default_export
register_global_func('default')(default_build)
register_global_func('default_export')(default_export)
import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.meta_schedule import tir_integration

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
TGT=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})

od='F:/tvm/experiments/data/test_prune';os.makedirs(od,exist_ok=True)
print('Testing PRUNING DLL...',flush=True)
t0=time.time()
try:
    db=tir_integration.tune_tir(mod=M,target=TGT,work_dir=od,max_trials_global=8,num_trials_per_iter=8,strategy='evolutionary',space='post-order-apply',num_tuning_cores=1)
    records=db.get_all_tuning_records()
    print(f'{len(records)} records in {time.time()-t0:.0f}s',flush=True)
except Exception as e:
    print(f'tune_tir FAIL: {e}')

log=f'{od}/logs/tvm.s_tir.meta_schedule.logging.task_0_main.log'
with open(log)as f:c=f.read()
idx=c.find('Sample-Init-Population summary')
rej=0
if idx>=0:
    m=re.search(r'VerifyGPUCode.*?(\d+) failure',c[idx:idx+800])
    if m:rej=int(m.group(1))
trace_fails=sum(int(s)for s in re.findall(r'Trace replay failures: (\d+)',c))
print(f'VerifyGPUCode rejections: {rej}')
print(f'Trace replay failures: {trace_fails}')
print(f'')
print(f'OLD (no pruning): VerifyGPUCode ~450, Trace fails ~0')
print(f'NEW (pruning):    VerifyGPUCode {rej}, Trace fails {trace_fails}')
if trace_fails>0:
    print(f'PRUNING WORKS! {trace_fails} candidates caught before postproc')
else:
    print(f'Pruning not triggering - API may need fix')
