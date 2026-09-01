"""
6 operators: MatMul 128/256/512, Conv2D, Reduction, ReLU.
Each: C++ tune_tir(max_trials=50) -> parse log -> top-K benchmark.
"""
import sys,os,time,json,re
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
from tvm.runtime._tensor import empty as tv_empty
from tvm.s_tir.meta_schedule import tir_integration

TGT=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})
SAVE='F:/tvm/experiments/data'

# ── Operators ──
@tvm.script.ir_module
class MM128:
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
class MM256:
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
class MM512:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(512,512),'float32');B=T.match_buffer(b,(512,512),'float32');C=T.match_buffer(c,(512,512),'float32')
        for i,j,k in T.grid(512,512,512):
            with T.sblock('m'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

@tvm.script.ir_module
class Conv2D:
    @T.prim_func(s_tir=True)
    def main(x:T.handle,w:T.handle,y:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        X=T.match_buffer(x,(1,3,224,224),'float32')
        W=T.match_buffer(w,(64,3,7,7),'float32')
        Y=T.match_buffer(y,(1,64,112,112),'float32')
        for n,oc,oh,ow,ic,kh,kw in T.grid(1,64,112,112,3,7,7):
            with T.sblock('c'):
                vn,voc,voh,vow,vic,vkh,vkw=T.axis.remap('SSSSRRR',[n,oc,oh,ow,ic,kh,kw])
                with T.init():Y[vn,voc,voh,vow]=T.float32(0)
                Y[vn,voc,voh,vow]=Y[vn,voc,voh,vow]+X[vn,vic,voh+vkh,vow+vkw]*W[voc,vic,vkh,vkw]

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
class ReLU128:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(128,128),'float32');B=T.match_buffer(b,(128,128),'float32')
        for i,j in T.grid(128,128):
            with T.sblock('l'):
                vi,vj=T.axis.remap('SS',[i,j])
                B[vi,vj]=T.max(A[vi,vj],T.float32(0))

ops=[
('MM128',MM128,128,128,128,'MatMul 128^3'),
('MM256',MM256,256,256,256,'MatMul 256^3'),
('MM512',MM512,512,512,512,'MatMul 512^3'),
('Conv2D',Conv2D,1,3,224,'Conv2D 64x3x7x7'),
('Red128',Reduction128,128,128,1,'Reduction 128'),
('ReLU128',ReLU128,128,128,1,'ReLU 128'),
]

for tag,mod,M,N,K,desc in ops:
    od=f'{SAVE}/op_{tag}';os.makedirs(od,exist_ok=True)
    sep='='*50;print(f'\n{sep}\n{desc}\n{sep}',flush=True)
    t0=time.time()
    try:
        db=tir_integration.tune_tir(mod=mod,target=TGT,work_dir=od,max_trials_global=50,num_trials_per_iter=8,strategy='evolutionary',space='post-order-apply',num_tuning_cores=1)
    except Exception as e:
        print(f'  TUNE FAIL: {e}');continue
    records=db.get_all_tuning_records();tune_t=time.time()-t0

    # Parse log
    log=f'{od}/logs/tvm.s_tir.meta_schedule.logging.task_0_main.log'
    init_rej=0;build_sends=0;runner_sends=0;build_errs=0
    with open(log)as f:
        c=f.read()
        idx=c.find('Sample-Init-Population summary')
        if idx>=0:
            m=re.search(r'VerifyGPUCode.*?(\d+) failure',c[idx:idx+800])
            if m:init_rej=int(m.group(1))
        build_sends=sum(int(s)for s in re.findall(r'Sending (\d+) sample\(s\) to builder',c))
        runner_sends=sum(int(s)for s in re.findall(r'Sending (\d+) valid.*runner',c))
        build_errs=sum(int(s)for s in re.findall(r'Build errors: (\d+)',c))
        ts=re.findall(r'(\d{2}:\d{2}:\d{2})',c)
        span=0
        if len(ts)>1:
            def to_sec(t):h,m,sec=map(int,t.split(':'));return h*3600+m*60+sec
            span=to_sec(ts[-1])-to_sec(ts[0])

    print(f'  Records:{len(records)} Tune:{tune_t:.0f}s Log:{span:.0f}s')
    print(f'  SampleInit reject:{init_rej} Builder:{build_sends} Runner:{runner_sends} BuildErr:{build_errs}')

    # Build & benchmark top 3
    dev=tvm.cuda(0)
    if K>1:
        a=np.random.randn(M,N).astype('float32');b=np.random.randn(N,K).astype('float32');ref=a@b
    elif desc.startswith('Red'):
        a=np.random.randn(M,N).astype('float32');b=np.zeros(1);ref=np.sum(a,axis=1)
    else:
        a=np.random.randn(M,N).astype('float32');b=np.zeros(1);ref=np.maximum(a,0)

    results=[]
    for i,rec in enumerate(records[:5]):
        try:
            trace_json=rec.trace.as_json()
            sch=Schedule(mod,debug_mask='all');TraceApplyJSONToSchedule(trace_json,sch)
            rt=tvm.build(sch.mod,target=TGT)
            if K>1:c=np.zeros((M,K),'float32')
            else:c=np.zeros((M,N),'float32')
            a_g=tv_empty(a.shape,'float32',dev);a_g.copyfrom(a)
            c_g=tv_empty(c.shape,'float32',dev);c_g.copyfrom(c)
            if K>1:
                b_g=tv_empty(b.shape,'float32',dev);b_g.copyfrom(b)
                te=rt.time_evaluator(rt.entry_name,dev,number=10,repeat=3)
                r=te(a_g,b_g,c_g)
            else:
                te=rt.time_evaluator(rt.entry_name,dev,number=10,repeat=3)
                r=te(a_g,c_g)
            err=np.max(np.abs(c_g.numpy()-ref))
            flops=2*M*N*K/(r.mean*1e9)if K>1 else 0
            results.append({'idx':i,'latency_ms':r.mean*1000,'gflops':flops,'error':float(err)})
        except Exception as e:results.append({'idx':i,'error':str(e)[:60]})
    results.sort(key=lambda x:x.get('latency_ms',999))
    print(f'  Top 3:')
    for r in results[:3]:
        if 'latency_ms'in r:
            idx=r['idx'];lat=r['latency_ms'];gf=r['gflops'];err=r['error']
            print(f'    #{idx}: {lat:.3f}ms {gf:.0f}GFLOPS err={err:.2e}')
        else:print(f'    #{r["idx"]}: FAIL')

    with open(f'{od}/summary.json','w')as f:json.dump({'desc':desc,'records':len(records),'tune_s':tune_t,'init_rej':init_rej,'builder':build_sends,'runner':runner_sends,'build_err':build_errs,'top':results},f,indent=2,default=str)

print('\nALL DONE')
