"""
Pruning vs Baseline comparison on MatMul 128 and 256.
"""
import sys,os,time,re,json
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
import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.meta_schedule import tir_integration

@tvm.script.ir_module
class M128:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(128,128),'float32');B=T.match_buffer(b,(128,128),'float32');C=T.match_buffer(c,(128,128),'float32')
        for i,j,k in T.grid(128,128,128):
            with T.sblock('m'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

@tvm.script.ir_module
class M256:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(256,256),'float32');B=T.match_buffer(b,(256,256),'float32');C=T.match_buffer(c,(256,256),'float32')
        for i,j,k in T.grid(256,256,256):
            with T.sblock('m'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

TGT=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})

def run_one(name,mod,n_runs=3):
    results=[]
    for r in range(n_runs):
        od=f'F:/tvm/experiments/data/final_{name}_{r}';os.makedirs(od,exist_ok=True)
        t0=time.time()
        try:
            db=tir_integration.tune_tir(mod=mod,target=TGT,work_dir=od,
                max_trials_global=8,num_trials_per_iter=8,
                strategy='evolutionary',space='post-order-apply',num_tuning_cores=1)
            records=db.get_all_tuning_records()
            elapsed=time.time()-t0
            log=od+'/logs/tvm.s_tir.meta_schedule.logging.task_0_main.log'
            sched=od+'/logs/tvm.s_tir.meta_schedule.logging.task_scheduler.log'
            init_rej=0;tfs=0;builder=0
            if os.path.exists(log):
                with open(log)as f:c=f.read()
                tfs=sum(int(s)for s in re.findall(r'Trace replay failures: (\d+)',c))
                idx=c.find('Sample-Init-Population summary')
                if idx>=0:
                    m=re.search(r'VerifyGPUCode.*?(\d+) failure',c[idx:idx+800])
                    if m:init_rej=int(m.group(1))
            if os.path.exists(sched):
                with open(sched)as f:c=f.read()
                builder=sum(int(s)for s in re.findall(r'Sending (\d+) sample',c))
            results.append({'records':len(records),'time':elapsed,'builder':builder,
                           'trace_fails':tfs,'verify_gpu_rej':init_rej})
            print(f'  {name} run{r}: rec={len(records)} time={elapsed:.0f}s build={builder} tfail={tfs} vgpu={init_rej}')
        except Exception as e:
            results.append({'error':str(e)[:100]})
            print(f'  {name} run{r}: FAIL {type(e).__name__}')
    return results

print('=== PRUNING DLL COMPARISON ===')
print(f'DLL: {os.path.getmtime("F:/tvm/build/lib/tvm_runtime.dll")}')
print()

for mod,name in [(M128,'M128'),(M256,'M256')]:
    print(f'{name}:')
    res=run_one(name,mod,n_runs=3)
    # Summary
    ok=[r for r in res if 'records' in r]
    if ok:
        avg_time=sum(r['time']for r in ok)/len(ok)
        avg_builder=sum(r['builder']for r in ok)/len(ok)
        avg_tf=sum(r['trace_fails']for r in ok)/len(ok)
        avg_vgpu=sum(r['verify_gpu_rej']for r in ok)/len(ok)
        print(f'  AVG: time={avg_time:.0f}s builder={avg_builder:.1f} tfail={avg_tf:.1f} vgpu={avg_vgpu:.0f}')
        if avg_tf>0:print(f'  *** PRUNING ACTIVE: {avg_tf:.0f} candidates caught before VerifyGPUCode!')
        elif avg_builder>0:print(f'  Builder got candidates - no pruning effect')
        else:print(f'  All blocked - pruning too aggressive')
    print()

with open('F:/tvm/experiments/data/final_compare.json','w')as f:json.dump(res,f,indent=2,default=str)
print('Saved: F:/tvm/experiments/data/final_compare.json')
