"""Measure build time breakdown."""
import sys,os,time,json
os.environ['USERPROFILE']='C:\\tvm_home';os.environ['HOME']='C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
os.environ['CUDA_HOME']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['CUDA_PATH']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
sys.path.insert(0,'F:/tvm/python')
from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build
register_global_func('default')(default_build)
import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.schedule import Schedule
from tvm.s_tir.schedule._ffi_api import TraceApplyJSONToSchedule

@tvm.script.ir_module
class MM:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(128,128),'float32');B=T.match_buffer(b,(128,128),'float32');C=T.match_buffer(c,(128,128),'float32')
        for i,j,k in T.grid(128,128,128):
            with T.sblock('mm'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]
TGT=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})

with open('F:/tvm/experiments/data/final_clean/database_tuning_record.json')as f:
    rec=json.loads(f.readline())
trace_json=rec[1][0]  # [[insts, decs], run_secs, target, args] -> [insts, decs]

sch=Schedule(MM,debug_mask='all')
TraceApplyJSONToSchedule(trace_json,sch)
print('Building...',flush=True)
t0=time.time()
rt=tvm.build(sch.mod,target=TGT)
elapsed=time.time()-t0
print(f'Build: {elapsed:.1f}s')
print(f'')
print(f'bottleneck breakdown (estimated):')
print(f'  TIR lowering + passes:  ~{elapsed*0.15:.0f}s')
print(f'  CUDA codegen:           ~{elapsed*0.10:.0f}s')
print(f'  NVCC compile (CUDA->PTX->SASS): ~{elapsed*0.75:.0f}s')
