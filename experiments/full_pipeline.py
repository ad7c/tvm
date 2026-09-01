"""
Full Meta Schedule pipeline: SampleInit + Evolve + Benchmark.
MatMul 128^3, max_trials=50, num_per_iter=8.
"""
import sys,os,time,json
os.environ['USERPROFILE']='C:\\tvm_home'
os.environ['HOME']='C:\\tvm_home'
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
OD='F:/tvm/experiments/data'
os.makedirs(OD,exist_ok=True)

# ── Phase 1: Full Meta Schedule tuning ──
from tvm.s_tir.meta_schedule import tir_integration
print('Phase 1: Full Meta Schedule (max_trials=50)...')
t0=time.time()
db=tir_integration.tune_tir(mod=MatMul128,target=TGT,work_dir=f'{OD}/full_tune2',
    max_trials_global=50,num_trials_per_iter=8,
    strategy='evolutionary',space='post-order-apply',num_tuning_cores=1)
records=db.get_all_tuning_records()
tune_time=time.time()-t0
print(f'Tuning done: {len(records)} records in {tune_time:.0f}s')

# ── Phase 2: Build & benchmark each valid record ──
print(f'\nPhase 2: Build & benchmark {min(10,len(records))} records...')
results=[]
for i,rec in enumerate(records[:10]):
    print(f'  Rec{i}:',end=' ',flush=True)
    try:
        trace_json=rec.trace.as_json()
        sch=Schedule(MatMul128,debug_mask='all')
        TraceApplyJSONToSchedule(trace_json,sch)
        rt=tvm.build(sch.mod,target=TGT)
        dev=tvm.cuda(0)
        from tvm.runtime._tensor import empty as tv_empty
        a=np.random.randn(128,128).astype('float32')
        b=np.random.randn(128,128).astype('float32')
        c=np.zeros((128,128),'float32')
        a_g=tv_empty((128,128),'float32',dev);a_g.copyfrom(a)
        b_g=tv_empty((128,128),'float32',dev);b_g.copyfrom(b)
        c_g=tv_empty((128,128),'float32',dev);c_g.copyfrom(c)
        te=rt.time_evaluator(rt.entry_name,dev,number=10,repeat=5)
        r=te(a_g,b_g,c_g)
        err=np.max(np.abs(c_g.numpy()-a@b))
        gflops=2*128**3/(r.mean*1e9)
        print(f'{r.mean*1000:.3f}ms {gflops:.0f}GFLOPS err={err:.2e}')
        results.append({'id':i,'latency_ms':r.mean*1000,'gflops':gflops,'error':float(err)})
    except Exception as e:
        print(f'FAIL: {type(e).__name__}')

# ── Save ──
summary={'records':len(records),'tune_time_s':tune_time,'results':results}
with open(f'{OD}/full_pipeline.json','w')as f:json.dump(summary,f,indent=2,default=str)
print(f'\nSaved: {OD}/full_pipeline.json')
if results:
    lats=[r['latency_ms']for r in results]
    print(f'Latency: min={min(lats):.3f}ms avg={sum(lats)/len(lats):.3f}ms best={lats[0]:.3f}ms')
    gflops_list=[r['gflops']for r in results]
    print(f'GFLOPS:  max={max(gflops_list):.0f} avg={sum(gflops_list)/len(gflops_list):.0f}')
