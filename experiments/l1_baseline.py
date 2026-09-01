"""L1 Baseline: replay tune_tir records -> build -> benchmark."""
import sys,os,time,json
os.environ['USERPROFILE']='C:\\tvm_home'
os.environ['HOME']='C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
os.environ['PYTHONPATH']='F:/tvm/python'
os.environ['CUDA_HOME']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['CUDA_PATH']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
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

@tvm.script.ir_module
class MatMul128:
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

# Load existing records
DB='F:/tvm/experiments/data/tune_demo2/database_tuning_record.json'
print(f'Loading {DB}...')
records=[]
with open(DB,'r') as f:
    for line in f:
        records.append(json.loads(line))
print(f'{len(records)} records loaded')

results=[]
for i,rec in enumerate(records[:3]):
    print(f'\nRecord {i}:',end=' ',flush=True)
    try:
        full=rec[1]  # [[insts,decs], run_secs, target, args]
        trace_json=full[0]  # [insts, decs]

        sch=Schedule(MatMul128,debug_mask='all')
        TraceApplyJSONToSchedule(trace_json,sch)
        rt=tvm.build(sch.mod,target=TGT)
        dev=tvm.cuda(0)

        a=np.random.randn(128,128).astype('float32')
        b=np.random.randn(128,128).astype('float32')
        c=np.zeros((128,128),'float32')
        from tvm.runtime._tensor import empty as tvm_empty
        a_g=tvm_empty((128,128),'float32',dev);a_g.copyfrom(a)
        b_g=tvm_empty((128,128),'float32',dev);b_g.copyfrom(b)
        c_g=tvm_empty((128,128),'float32',dev);c_g.copyfrom(c)
        te=rt.time_evaluator(rt.entry_name,dev,number=10,repeat=5)
        r=te(a_g,b_g,c_g)
        err=np.max(np.abs(c_g.numpy()-a@b))
        gflops=2*128**3/(r.mean*1e9)
        print(f'{r.mean*1000:.3f}ms {gflops:.0f}GFLOPS err={err:.2e}')
        results.append({'id':i,'latency_ms':r.mean*1000,'gflops':gflops,'error':float(err)})
    except Exception as e:
        print(f'FAIL: {type(e).__name__}: {str(e)[:120]}')

if results:
    lats=[r['latency_ms']for r in results]
    print(f'\nLatency: min={min(lats):.3f}ms avg={sum(lats)/len(lats):.3f}ms max={max(lats):.3f}ms')
    with open('F:/tvm/experiments/data/l1_baseline.json','w') as f:
        json.dump({'records':len(records),'results':results},f,indent=2)
    print('Saved: F:/tvm/experiments/data/l1_baseline.json')
