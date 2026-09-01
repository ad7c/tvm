"""Parse tune_tir log: SampleInit rejection count, time."""
import sys,os,re,json
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
import tvm,time
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.meta_schedule import tir_integration

@tvm.script.ir_module
class M:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(128,128),'float32');B=T.match_buffer(b,(128,128),'float32');C=T.match_buffer(c,(128,128),'float32')
        for i,j,k in T.grid(128,128,128):
            with T.sblock('mm'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]
tgt=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})

od='F:/tvm/experiments/data/parse_meas'
os.makedirs(od,exist_ok=True)
print('Running tune_tir (max_trials=50)...')
t0=time.time()
db=tir_integration.tune_tir(mod=M,target=tgt,work_dir=od,
    max_trials_global=50,num_trials_per_iter=8,
    strategy='evolutionary',space='post-order-apply',num_tuning_cores=1)
records=db.get_all_tuning_records()
total_time=time.time()-t0
print(f'Total: {len(records)} records, {total_time:.0f}s')

# Parse the main log file
log=od+'/logs/tvm.s_tir.meta_schedule.logging.task_0_main.log'
with open(log)as f:
    content=f.read()

# Find Sample-Init-Population summary with timestamp
# Format: "YYYY-MM-DD HH:MM:SS [INFO] [evolutionary_search.cc:551] Sample-Init-Population summary:"
# Followed by postproc failure lines
# Followed by "Sending X sample(s) to builder"

init_failures=0
init_lines=[]
lines=content.split('\n')
in_init_summary=False
for i,line in enumerate(lines):
    if 'Sample-Init-Population summary' in line:
        in_init_summary=True
        init_lines.append(line)
        continue
    if in_init_summary:
        init_lines.append(line)
        if 'VerifyGPUCode' in line:
            m=re.search(r'(\d+) failure',line)
            if m:init_failures+=int(m.group(1))
        if 'Sending' in line and 'builder' in line:
            in_init_summary=False
            break

# Find timestamps
first_ts=None;last_ts=None
ts_pattern=re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
matches=ts_pattern.findall(content)
if matches:
    first_ts=matches[0];last_ts=matches[-1]

print(f'\nSampleInit VerifyGPUCode rejections: {init_failures}')
print(f'Tuning start: {first_ts}, end: {last_ts}')
print(f'Total time: {total_time:.0f}s')
print(f'\nSummary: to find {len(records)} valid candidates,')
print(f'  SampleInit rejected ~{init_failures} candidates at VerifyGPUCode')
if len(records)>0:
    print(f'  Per valid: ~{init_failures/len(records):.0f} rejections in init phase')
