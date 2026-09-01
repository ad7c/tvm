"""Build & benchmark ALL 50 records, find top-K."""
import sys,os,json
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
import tvm,numpy as np
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.schedule import Schedule
from tvm.s_tir.schedule._ffi_api import TraceApplyJSONToSchedule
from tvm.runtime._tensor import empty as tv_empty

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

# Load ALL records
db_file='F:/tvm/experiments/data/full_tune/database_tuning_record.json'
records=[]
with open(db_file)as f:
    for line in f:records.append(json.loads(line))
print(f'Loaded {len(records)} records')

# Build & benchmark ALL
dev=tvm.cuda(0)
a=np.random.randn(128,128).astype('float32')
b=np.random.randn(128,128).astype('float32')
ref=a@b
all_results=[]

for i,rec in enumerate(records):
    try:
        full=rec[1];trace_json=full[0]
        sch=Schedule(MM,debug_mask='all')
        TraceApplyJSONToSchedule(trace_json,sch)
        rt=tvm.build(sch.mod,target=TGT)
        c=np.zeros((128,128),'float32')
        a_g=tv_empty((128,128),'float32',dev);a_g.copyfrom(a)
        b_g=tv_empty((128,128),'float32',dev);b_g.copyfrom(b)
        c_g=tv_empty((128,128),'float32',dev);c_g.copyfrom(c)
        te=rt.time_evaluator(rt.entry_name,dev,number=10,repeat=3)
        r=te(a_g,b_g,c_g)
        err=np.max(np.abs(c_g.numpy()-ref))
        all_results.append({'idx':i,'latency_ms':r.mean*1000,'gflops':2*128**3/(r.mean*1e9),'error':float(err)})
    except Exception as e:
        all_results.append({'idx':i,'error_str':f'{type(e).__name__}: {str(e)[:60]}'})

# Sort by latency
all_results.sort(key=lambda x:x.get('latency_ms',999))
top_n=5

print(f'\nTop {top_n} candidates (by latency):')
for r in all_results[:top_n]:
    print(f'  #{r["idx"]:2d}: {r["latency_ms"]:.3f}ms  {r["gflops"]:.0f}GFLOPS  err={r["error"]:.2e}')

# Save
with open('F:/tvm/experiments/data/top_k.json','w')as f:json.dump({'all':all_results,'top':all_results[:top_n]},f,indent=2)
print(f'\nBest: #{all_results[0]["idx"]} at {all_results[0]["latency_ms"]:.3f}ms ({all_results[0]["gflops"]:.0f}GFLOPS)')
print(f'Saved: F:/tvm/experiments/data/top_k.json')
