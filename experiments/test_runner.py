"""Test runner — single vs multi tune_tir calls."""
import sys,os,time
os.environ['USERPROFILE']='C:\\tvm_home';os.environ['HOME']='C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
os.environ['CUDA_HOME']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['CUDA_PATH']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['PYTHONPATH']='F:/tvm/python'
# Use project-local temp to avoid Windows system temp lock
os.makedirs('F:/tvm/experiments/data/tmp',exist_ok=True)
os.environ['TMP']='F:\\tvm\\experiments\\data\\tmp'
os.environ['TEMP']='F:\\tvm\\experiments\\data\\tmp'
os.environ['TMPDIR']='F:\\tvm\\experiments\\data\\tmp'
sys.path.insert(0,'F:/tvm/python')
from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build,default_export
register_global_func('default')(default_build)
register_global_func('default_export')(default_export)
import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.meta_schedule import tir_integration
import gc

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

for i in range(1):
    od=f'F:/tvm/experiments/data/test_runner_50'
    os.makedirs(od,exist_ok=True)
    print(f'Test max_trials=50:',flush=True)
    t0=time.time()
    try:
        db=tir_integration.tune_tir(mod=M,target=TGT,work_dir=od,
            max_trials_global=50,num_trials_per_iter=8,
            strategy='evolutionary',space='post-order-apply',num_tuning_cores=1)
        records=db.get_all_tuning_records()
        print(f'  OK: {len(records)} records in {time.time()-t0:.0f}s')
    except Exception as e:
        print(f'  FAIL: {type(e).__name__}: {str(e)[:200]}')
    pass
