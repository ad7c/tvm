"""
Run C++ pruning experiments: compare with/without pruning.
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

TGT=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})

print('=== C++ Pruning Test ===')
for run in range(3):
    od='F:/tvm/experiments/data/prune_run_'+str(run)
    os.makedirs(od,exist_ok=True)
    print('Run '+str(run)+':',end=' ',flush=True)
    t0=time.time()
    try:
        db=tir_integration.tune_tir(mod=M128,target=TGT,work_dir=od,
            max_trials_global=8,num_trials_per_iter=8,
            strategy='evolutionary',space='post-order-apply',num_tuning_cores=1)
        records=db.get_all_tuning_records();elapsed=time.time()-t0
        log=od+'/logs/tvm.s_tir.meta_schedule.logging.task_0_main.log'
        sched=od+'/logs/tvm.s_tir.meta_schedule.logging.task_scheduler.log'
        rej=0;tfs=0;builder=0
        if os.path.exists(log):
            with open(log)as f:c=f.read()
            tfs=sum(int(s)for s in re.findall(r'Trace replay failures: (\d+)',c))
            idx=c.find('Sample-Init-Population summary')
            if idx>=0:
                m=re.search(r'VerifyGPUCode.*?(\d+) failure',c[idx:idx+800])
                if m:rej=int(m.group(1))
        if os.path.exists(sched):
            with open(sched)as f:c=f.read()
            builder=sum(int(s)for s in re.findall(r'Sending (\d+) sample',c))
        print('rec='+str(len(records))+' time='+str(int(elapsed))+'s build='+str(builder)+' tfail='+str(tfs)+' vgpu='+str(rej))
        status='PRUNING WORKS' if tfs>0 else ('BLOCKED ALL' if builder==0 else 'not pruning')
        print('  '+status)
    except Exception as e:
        print('FAIL '+type(e).__name__)
