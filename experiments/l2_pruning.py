"""
L2: Space analysis + empirical pruning comparison.
Part A: Theoretical pruning ratio (divisor enumeration)
Part B: Manual sampling with/without prefilter, measure TVM calls saved
"""
import sys,os,time,random,json,math
from collections import Counter

os.environ['USERPROFILE']='C:\\tvm_home'
os.environ['HOME']='C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
os.environ['CUDA_HOME']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['CUDA_PATH']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
sys.path.insert(0,'F:/tvm/python')

from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build,default_export
register_global_func('default')(default_build)
register_global_func('default_export')(default_export)

import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.schedule import Schedule,Trace
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply
from tvm.s_tir.meta_schedule.postproc import Postproc

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

TGT=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})
GPU={'shmem':49152,'threads_sm':1536,'shmem_sm':102400}

# ═══ Part A: Theoretical space analysis ═══
def divisors(n):
    return sorted({d for i in range(1,int(math.isqrt(n))+1)if n%i==0 for d in(i,n//i)})

print('='*60)
print('Part A: Theoretical Pruning Analysis (MatMul 128^3)')
print('='*60)

divs=divisors(128); n_divs=len(divs)
total_block=0; upward_pass=0; dual_pass=0

for bi in divs:
    ti=128//bi
    for bj in divs:
        tj=128//bj; threads=ti*tj
        if threads<32 or threads>1024:continue
        for bk in divs:
            total_block+=1
            sh=(bi*bk+bk*bj)*8+bi*bj*4
            # Upward
            if sh>GPU['shmem']:continue
            upward_pass+=1
            # Downward
            occ_chk=min(GPU['threads_sm']//max(threads,1),GPU['shmem_sm']//max(sh,1))
            occ=max(1,occ_chk)*threads/GPU['threads_sm']
            if occ<0.25:continue
            flops=2*bi*bj*bk;mem=(bi*bk+bk*bj)*4
            if mem>0 and flops/mem<8.0:continue
            dual_pass+=1

print(f'Divisors: {n_divs}')
print(f'Total block tile combos (threads filter): {total_block}')
print(f'After UPWARD (shmem):    {upward_pass} ({upward_pass/total_block*100:.1f}%)')
print(f'After DUAL (occ+reuse): {dual_pass} ({dual_pass/total_block*100:.1f}%)'if dual_pass else '')
print(f'Upward prune rate: {(1-upward_pass/total_block)*100:.1f}%')
if dual_pass:print(f'Dual prune rate: {(1-dual_pass/total_block)*100:.1f}%')

# ═══ Part B: Empirical — manual sampling, measure TVM calls saved ═══
print(f'\n{"="*60}')
print('Part B: Empirical Sampling (max 500 attempts per mode)')
print('='*60)

sg=PostOrderApply();ctx=TuneContext(mod=M,target=TGT,space_generator=sg,task_name='l2')
ctx.space_generator._initialize_with_tune_context(ctx)
ds=ctx.space_generator.generate_design_space(M)
insts=[d.trace.insts for d in ds]
pps=[pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]
print(f'DS:{len(insts)} PP:{len(pps)}')

def extract_tile(trace):
    dm=trace.decisions;sp,rd=[],[]
    for k,v in list(dm.items()):
        if 'sample_perfect_tile'not in str(k).lower():continue
        try:vs=tuple(int(vv.value)if hasattr(vv,'value')else int(vv)for vv in list(v))
        except:continue
        (sp if'n=5'in str(k)else rd).append(vs)
    if len(sp)<2:return None
    si,sj=sp[0],sp[1];bi,ti=si[0],si[2];bj,tj=sj[0],sj[2]
    bk=rd[0][0]if rd else 0;vk=rd[0][1]if rd else None
    return{'bi':bi,'bj':bj,'bk':bk,'threads':ti*tj,'vk':vk}

def upward_prune(info):
    if info is None:return False
    b,t=info['bi'],info['threads']
    if t<32 or t>1024:return False
    sh=(b*info['bk']+info['bk']*info['bj'])*8+b*info['bj']*4
    if sh>GPU['shmem']:return False
    if info['vk']==1:return False
    return True

def dual_prune(info):
    """向上硬砍 + 向下只硬砍 occupancy. 数据复用度只记录不砍."""
    if not upward_prune(info):return False
    bi,bj,bk,t=info['bi'],info['bj'],info['bk'],info['threads']
    sh=(bi*bk+bk*bj)*8+bi*bj*4
    b_thr=GPU['threads_sm']//max(t,1);b_sh=GPU['shmem_sm']//max(sh,1)
    occ=max(1,min(b_thr,b_sh))*t/GPU['threads_sm']
    if occ<0.20:return False  # lowered from 25% for smaller GPU
    # data reuse: penalty only, hard cutoff removed
    return True

MAX=500
results={}
for label,pf in[('baseline',None),('upward',upward_prune),('dual',dual_prune)]:
    random.seed(42);t0=time.time()
    tvm_calls=0;pruned=0;valid=0
    for a in range(1,MAX+1):
        t=Trace(insts[random.randint(0,2)],{})
        try:
            sch=Schedule(M,debug_mask='all');t.apply_to_schedule(sch,remove_postproc=False)
            info=extract_tile(sch.trace)
            if pf and info and not pf(info):
                pruned+=1;continue
            tvm_calls+=1
            sch.enter_postproc();ok=True
            for pp in pps:
                if not pp.apply(sch):ok=False;break
            if ok:valid+=1
        except:pass
    elapsed=time.time()-t0
    results[label]={'attempts':MAX,'tvm_calls':tvm_calls,'pruned':pruned,'valid':valid,'time':elapsed}
    print(f'  {label:10s}: {MAX} attempts, {tvm_calls} TVM calls, {pruned} pruned ({pruned/MAX*100:.0f}%), {valid} valid, {elapsed:.0f}s')

# Save
with open('F:/tvm/experiments/data/l2_pruning.json','w')as f:json.dump(results,f,indent=2)
print(f'Saved: F:/tvm/experiments/data/l2_pruning.json')
