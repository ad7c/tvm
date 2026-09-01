"""
Clean run: C++ tune_tir -> parse log -> build top-K -> GPU benchmark.
Single script, all data fresh, no cached results.
"""
import sys,os,time,json,re
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

OD='F:/tvm/experiments/data/final_clean'
os.makedirs(OD,exist_ok=True)

# ── Phase 1: tune ──
print('Phase 1: tune_tir (max_trials=50)...',flush=True)
from tvm.s_tir.meta_schedule import tir_integration
t0=time.time()
db=tir_integration.tune_tir(mod=MM,target=TGT,work_dir=OD,
    max_trials_global=50,num_trials_per_iter=8,
    strategy='evolutionary',space='post-order-apply',num_tuning_cores=1)
records=db.get_all_tuning_records()
tune_time=time.time()-t0
print(f'Tuning done: {len(records)} records in {tune_time:.0f}s')

# ── Phase 2: parse log ──
log_file=f'{OD}/logs/tvm.s_tir.meta_schedule.logging.task_0_main.log'
init_rejections=0
events=[]
with open(log_file)as f:
    content=f.read()
    # SampleInit rejections
    idx=content.find('Sample-Init-Population summary')
    if idx>=0:
        m=re.search(r'VerifyGPUCode.*?(\d+) failure',content[idx:idx+800])
        if m:init_rejections=int(m.group(1))
    # All events with timestamps
    for line in content.split('\n'):
        m=re.search(r'(\d{2}:\d{2}:\d{2}).*?(Sending \d+ sample|Build errors|runner|Task #0|Sample-Init)',line)
        if m:events.append((m.group(1),m.group(2)))
    # Time range
    all_ts=re.findall(r'(\d{2}:\d{2}:\d{2})',content)
    def s(t):h,m,sec=map(int,t.split(':'));return h*3600+m*60+sec
    span=s(all_ts[-1])-s(all_ts[0])if len(all_ts)>1 else 0

print(f'\nPhase 2: Log analysis')
print(f'  Time span: {span}s ({span/60:.1f}min)')
print(f'  SampleInit VerifyGPUCode rejections: {init_rejections}')
print(f'  Events:')
for ts,evt in events:print(f'    {ts} {evt}')

# ── Phase 3: build & benchmark ALL records, find top ──
print(f'\nPhase 3: Build & benchmark {len(records)} records...')
dev=tvm.cuda(0)
a=np.random.randn(128,128).astype('float32')
b=np.random.randn(128,128).astype('float32')
ref=a@b
all_results=[]

for i,rec in enumerate(records):
    try:
        trace_json=rec.trace.as_json()
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
        all_results.append({'idx':i,'error':str(e)[:80]})

# Sort by latency
all_results.sort(key=lambda x:x.get('latency_ms',999))

print(f'\nTop 5 (by latency):')
for r in all_results[:5]:
    if 'latency_ms' in r:
        print(f'  #{r["idx"]:2d}: {r["latency_ms"]:.3f}ms  {r["gflops"]:.0f}GFLOPS  err={r["error"]:.2e}')
    else:
        print(f'  #{r["idx"]:2d}: FAIL {r["error"][:60]}')

# ── Save ──
summary={
    'tuning':{'time_s':tune_time,'records':len(records),'span_log_s':span,
              'init_rejections':init_rejections,'init_attempts':init_rejections+50 if init_rejections>0 else 0,
              'events':events},
    'benchmark':all_results
}
with open(f'{OD}/summary.json','w')as f:json.dump(summary,f,indent=2,default=str)

print(f'\n=== BASELINE SUMMARY ===')
print(f'Tuning: {tune_time:.0f}s, {len(records)} records')
if init_rejections>0:
    print(f'SampleInit: {init_rejections} VerifyGPUCode rejections to find 50 initial')
    print(f'  Rejection rate: {init_rejections/(init_rejections+50)*100:.1f}%')
best=all_results[0]
if 'latency_ms' in best:
    print(f'Best: {best["latency_ms"]:.3f}ms, {best["gflops"]:.0f}GFLOPS')
print(f'Saved: {OD}/summary.json')
