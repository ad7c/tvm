"""
Multi-operator baseline: C++ tune_tir -> parse log -> top-K benchmark.
"""
import sys,os,time,json,re
os.environ['USERPROFILE']='C:\\tvm_home';os.environ['HOME']='C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
os.environ['CUDA_HOME']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['CUDA_PATH']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['PYTHONPATH']='F:/tvm/python'
# Fix WinError 5: use custom temp dir that subprocesses can access
os.makedirs('C:\\tvm_home\\tmp',exist_ok=True)
os.environ['TMP']='C:\\tvm_home\\tmp'
os.environ['TEMP']='C:\\tvm_home\\tmp'
os.environ['TMPDIR']='C:\\tvm_home\\tmp'
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
from tvm.s_tir.meta_schedule import tir_integration

TGT=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})
SAVE='F:/tvm/experiments/data'

# ── Operators ──
@tvm.script.ir_module
class MatMul128:
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
class MatMul256:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(256,256),'float32');B=T.match_buffer(b,(256,256),'float32');C=T.match_buffer(c,(256,256),'float32')
        for i,j,k in T.grid(256,256,256):
            with T.sblock('m'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

@tvm.script.ir_module
class Reduction128:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(128,128),'float32');B=T.match_buffer(b,(128,),'float32')
        for i,j in T.grid(128,128):
            with T.sblock('r'):
                vi,vj=T.axis.remap('SS',[i,j])
                with T.init():B[vi]=T.float32(0)
                B[vi]=B[vi]+A[vi,vj]

@tvm.script.ir_module
class Elem128:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(128,128),'float32');B=T.match_buffer(b,(128,128),'float32');C=T.match_buffer(c,(128,128),'float32')
        for i,j in T.grid(128,128):
            with T.sblock('e'):
                vi,vj=T.axis.remap('SS',[i,j])
                C[vi,vj]=A[vi,vj]+B[vi,vj]

ops=[('MatMul128',MatMul128,128,128,128),('MatMul256',MatMul256,256,256,256),('Reduction128',Reduction128,128,128,1),('Elem128',Elem128,128,128,1)]

for name,mod,M,N,K in ops:
    od=f'{SAVE}/multi_{name}';os.makedirs(od,exist_ok=True)
    sep='='*60;print(f'\n{sep}\n{name}\n{sep}',flush=True)

    # Phase 1: tune
    print('Tuning...',flush=True)
    t0=time.time()
    try:
        db=tir_integration.tune_tir(mod=mod,target=TGT,work_dir=od,max_trials_global=50,num_trials_per_iter=8,strategy='evolutionary',space='post-order-apply',num_tuning_cores=1)
    except Exception as e:
        print(f'  TUNE FAIL: {e}');continue
    records=db.get_all_tuning_records();tune_t=time.time()-t0

    # Phase 2: parse log
    log=f'{od}/logs/tvm.s_tir.meta_schedule.logging.task_0_main.log'
    init_rej=0;events=[]
    with open(log)as f:
        content=f.read()
        idx=content.find('Sample-Init-Population summary')
        if idx>=0:
            m=re.search(r'VerifyGPUCode.*?(\d+) failure',content[idx:idx+800])
            if m:init_rej=int(m.group(1))
        for line in content.split('\n'):
            m2=re.search(r'(\d{2}:\d{2}:\d{2}).*?(Sending \d+ sample|Build errors|runner)',line)
            if m2:events.append((m2.group(1),m2.group(2)))
        all_ts=re.findall(r'(\d{2}:\d{2}:\d{2})',content)
        def s(t):h,m,sec=map(int,t.split(':'));return h*3600+m*60+sec
        span=s(all_ts[-1])-s(all_ts[0])if len(all_ts)>1 else 0

    print(f'  Records: {len(records)}, Tune time: {tune_t:.0f}s, Log span: {span:.0f}s')
    print(f'  SampleInit VerifyGPUCode rejections: {init_rej}')
    print(f'  Events: {len(events)} rounds')

    # Phase 3: benchmark top 3
    dev=tvm.cuda(0)
    a=np.random.randn(M,N).astype('float32');b=np.random.randn(N,K).astype('float32')
    ref=a@b if K>1 else np.sum(a,axis=1)
    results=[]
    for i,rec in enumerate(records[:5]):
        try:
            trace_json=rec.trace.as_json()
            sch=Schedule(mod,debug_mask='all');TraceApplyJSONToSchedule(trace_json,sch)
            rt=tvm.build(sch.mod,target=TGT)
            c=np.zeros((M,K)if K>1 else(M,),'float32')
            a_g=tv_empty(a.shape,'float32',dev);a_g.copyfrom(a)
            b_g=tv_empty(b.shape,'float32',dev);b_g.copyfrom(b)
            c_g=tv_empty(c.shape,'float32',dev);c_g.copyfrom(c)
            te=rt.time_evaluator(rt.entry_name,dev,number=10,repeat=3)
            r=te(a_g,b_g,c_g) if K>1 else te(a_g,c_g)
            err=np.max(np.abs(c_g.numpy()-ref))
            flops=2*M*N*K/(r.mean*1e9)if K>1 else 0
            results.append({'idx':i,'latency_ms':r.mean*1000,'gflops':flops,'error':float(err)})
        except Exception as e:results.append({'idx':i,'error':str(e)[:60]})

    results.sort(key=lambda x:x.get('latency_ms',999))
    print(f'  Top 5:')
    for r in results[:5]:
        if 'latency_ms' in r:
            idx=r['idx'];lat=r['latency_ms'];gf=r['gflops'];err=r['error']
            print(f'    #{idx}: {lat:.3f}ms {gf:.0f}GFLOPS err={err:.2e}')
        else:print(f'    #{r["idx"]}: FAIL')

    # Save
    with open(f'{od}/summary.json','w')as f:json.dump({'name':name,'records':len(records),'tune_time_s':tune_t,'init_rejections':init_rej,'top':results},f,indent=2,default=str)

print('\nALL DONE')
