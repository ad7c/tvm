"""
Auto-experiment runner for bidirectional pruning research.
Runs all experiments sequentially, saves data, produces summary.
"""
import sys, os, time, random, json, re, math
from collections import Counter, defaultdict

os.environ['USERPROFILE']='C:\\tvm_home'; os.environ['HOME']='C:\\tvm_home'
sys.path.insert(0,'F:/tvm/python'); os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'

from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build, default_export
register_global_func('default')(default_build)
register_global_func('default_export')(default_export)

import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.schedule import Schedule
from tvm.s_tir.schedule.trace import Trace
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply
from tvm.s_tir.meta_schedule.postproc import Postproc


# ═══════════════════════════════════════════════════════════
# Shared setup
# ═══════════════════════════════════════════════════════════

@tvm.script.ir_module
class Matmul128:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({'global_symbol': 'main'})
        A = T.match_buffer(a, (128, 128), 'float32')
        B = T.match_buffer(b, (128, 128), 'float32')
        C = T.match_buffer(c, (128, 128), 'float32')
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock('matmul'):
                vi, vj, vk = T.axis.remap('SSR', [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

TARGET=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})
GPU={'shmem':49152,'threads':1024,'threads_sm':1536,'warp':32}
OUTDIR='F:/tvm/experiments/data'

def extract_tile_params(trace):
    dm=trace.decisions;sp,rd=[],[]
    for key,val in list(dm.items()):
        if 'sample_perfect_tile' not in str(key).lower(): continue
        try: vs=tuple(int(v.value)if hasattr(v,'value')else int(v)for v in list(val))
        except: continue
        if 'n=5' in str(key): sp.append(vs)
        elif 'n=3' in str(key): rd.append(vs)
    if len(sp)<2: return None,None,None,None,None,None
    si,sj=sp[0],sp[1];bi,ti=si[0],si[2];bj,tj=sj[0],sj[2]
    bk=rd[0][0]if rd else None;threads=ti*tj;vk=rd[0][1]if rd else None
    return bi,bj,bk,threads,vk,{'bi':bi,'bj':bj,'bk':bk,'threads':threads,'vk':vk,'si':si,'sj':sj,'rd':rd[0]if rd else None}

def est_shmem(bi,bj,bk): return (bi*bk+bk*bj)*8+bi*bj*4

def occupancy(threads,shmem):
    b_threads=GPU['threads_sm']//max(threads,1);b_shmem=102400//max(shmem,1)
    return max(1,min(b_threads,b_shmem))*threads/GPU['threads_sm']

def prefilter_single(info):
    if info is None: return False
    bi,bj,bk,threads,vk=info[0],info[1],info[2],info[3],info[4]
    if threads and (threads<32 or threads>1024): return False
    if bk and bj and est_shmem(bi,bj,bk)>GPU['shmem']: return False
    # vk=1 means cooperative fetch buffer leaks to outer loop → guaranteed VerifyGPUCode fail
    if vk is not None and vk==1: return False
    return True

def prefilter_double(info):
    if not prefilter_single(info): return False
    bi,bj,bk,threads=info[0],info[1],info[2],info[3]
    if bk and bj:
        shmem=est_shmem(bi,bj,bk)
        if shmem>0 and occupancy(threads,shmem)<0.25: return False
        flops=2*bi*bj*bk;mem=(bi*bk+bk*bj)*4
        if mem>0 and flops/mem<8.0: return False
    return True


# ═══════════════════════════════════════════════════════════
# Exp-1a: Search space analysis (pure arithmetic, fast)
# ═══════════════════════════════════════════════════════════

def exp1a_search_space():
    print('\n'+'='*60)
    print('Exp-1a: Search Space Analysis')
    print('='*60)
    results={}
    for extent,label in [(128,'128^3'),(256,'256^3')]:
        divs=sorted(set(d for i in range(1,int(math.isqrt(extent))+1)if extent%i==0 for d in (i,extent//i)))
        n_divs=len(divs)

        # Block tile space
        block_total=0;block_legal=0
        legal_block_tiles=[]
        for bi in divs:
            ti=extent//bi
            for bj in divs:
                tj=extent//bj;threads=ti*tj
                if threads<32 or threads>1024: continue
                for bk in divs:
                    block_total+=1
                    shmem=est_shmem(bi,bj,bk)
                    if shmem<=GPU['shmem']:
                        block_legal+=1
                        legal_block_tiles.append((bi,bj,bk,threads,shmem))

        # Full factor space (stars-and-bars)
        def count_combos(ext,n):
            return math.comb(ext+n-1,n-1) if ext>0 else 1
        n_spatial_5=count_combos(8 if extent==256 else 7,5)
        n_spatial_full=n_spatial_5**2
        n_reduction_3=count_combos(8 if extent==256 else 7,3)
        total_factor=n_spatial_full*n_reduction_3

        results[label]={
            'extent':extent,'n_divisors':n_divs,
            'block_total':block_total,'block_legal':block_legal,
            'block_legal_rate':block_legal/block_total*100 if block_total else 0,
            'spatial_combos': n_spatial_5,
            'reduction_combos': n_reduction_3,
            'total_factor_space':total_factor,
            'legal_tiles':legal_block_tiles[:20]
        }

        print(f'\n  MatMul {label}:')
        print(f'    Divisors: {n_divs}, Block tiles: {block_total}')
        print(f'    Block tiles legal: {block_legal} ({block_legal/block_total*100:.1f}%)')
        print(f'    Spatial combos (n=5): {n_spatial_5}')
        print(f'    Reduction combos (n=3): {n_reduction_3}')
        print(f'    Total factor space: {total_factor:,}')
        r_block=(1-block_legal/block_total)*100 if block_total else 0
        print(f'    Block-tile rejection rate: {r_block:.1f}%')

    with open(f'{OUTDIR}/exp1a_space.json','w')as f: json.dump(results,f,indent=2)
    return results


# ═══════════════════════════════════════════════════════════
# Exp-2: Three-mode comparison
# ═══════════════════════════════════════════════════════════

def run_one_mode(label,prefilter_fn,max_attempts,n_needed,seed):
    sg=PostOrderApply();ctx=TuneContext(mod=Matmul128,target=TARGET,space_generator=sg,task_name=f'exp2_{label}')
    ctx.space_generator._initialize_with_tune_context(ctx)
    ds=ctx.space_generator.generate_design_space(Matmul128)
    insts=[d.trace.insts for d in ds]

    postprocs=Postproc.create('cuda');skip={'RewriteTensorize','RewriteUnboundBlock'}
    pps=[pp for pp in postprocs if type(pp).__name__ not in skip]

    random.seed(seed)
    valid_at=[];valid_info=[];stats=Counter();t0=time.time()

    for attempt in range(1,max_attempts+1):
        ds_idx=random.randint(0,2);trace=Trace(insts[ds_idx],{})
        try:
            sch=Schedule(Matmul128,debug_mask='all')
            trace.apply_to_schedule(sch,remove_postproc=False)

            info=extract_tile_params(sch.trace)
            if prefilter_fn and info is not None:
                if not prefilter_fn(info):
                    stats['pruned']+=1;continue
                stats['prefilter_pass']+=1

            sch.enter_postproc();ok=True;bad=None
            for pp in pps:
                if not pp.apply(sch):ok=False;bad=type(pp).__name__;break

            if ok:
                valid_at.append(attempt)
                info_dict=info[5]if info else{}
                info_dict['attempt']=attempt
                valid_info.append(info_dict)
                stats['passed']+=1
                elapsed=time.time()-t0
                print(f'  [{attempt:5d}] VALID #{len(valid_at)}! {elapsed:.0f}s bi={info_dict.get("bi")} bj={info_dict.get("bj")} bk={info_dict.get("bk")} threads={info_dict.get("threads")} vk={info_dict.get("vk")}',flush=True)
                if len(valid_at)>=n_needed:break
            else:
                stats[f'tvm_{bad}']+=1
                # Print first 10 failures per mode for diagnostics
                if attempt<=10:
                    info_dict=info[5]if info else{}
                    print(f'  [{attempt:5d}] FAIL {bad} | bi={info_dict.get("bi")} bj={info_dict.get("bj")} bk={info_dict.get("bk")} threads={info_dict.get("threads")} vk={info_dict.get("vk")}',flush=True)
        except Exception as e:
            stats[f'exc_{type(e).__name__}']+=1

        if attempt%500==0:
            elapsed=time.time()-t0
            print(f'  [{attempt:5d}] {elapsed:.0f}s found={len(valid_at)} pruned={stats.get("pruned",0)} tvm_fails={sum(v for k,v in stats.items() if k.startswith("tvm_"))}',flush=True)

    elapsed=time.time()-t0
    total_pruned=stats.get('pruned',0)
    tvm_calls=attempt-total_pruned-len(valid_at)

    return {'label':label,'attempts':attempt,'valid':len(valid_at),
            'valid_at':valid_at,'valid_info':valid_info,
            'time_s':elapsed,'pruned':total_pruned,'tvm_calls':tvm_calls,
            'stats':dict(stats)}


def exp2_comparison():
    print('\n'+'='*60)
    print('Exp-2: Three-mode Comparison (MatMul 128^3)')
    print('Target: 3 valid per mode, max 5000 attempts')
    print('='*60)

    MAX=5000;NEED=3;results={}
    for mode, fn, desc in [
        ('baseline',None,'BASELINE: no pruning'),
        ('single',prefilter_single,'SINGLE: upward (shmem+threads)'),
        ('double',prefilter_double,'DOUBLE: upward+downward (occ+reuse)')
    ]:
        print(f'\n--- {desc} ---')
        r=run_one_mode(mode,fn,MAX,NEED,seed=42)
        results[mode]=r

    # Summary table
    print(f'\n{"="*60}')
    print(f'Exp-2 COMPARISON')
    print(f'{"="*60}')
    print(f'  {"Metric":30s} {"Baseline":>10s} {"Single":>10s} {"Double":>10s}')
    print(f'  {"-"*62}')
    for k,label in [('attempts','Total attempts'),('valid','Valid found'),
                     ('time_s','Time (s)'),('tvm_calls','TVM calls'),('pruned','Pruned')]:
        vals=[str(results[m][k])for m in['baseline','single','double']]
        print(f'  {label:30s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}')

    with open(f'{OUTDIR}/exp2_comparison.json','w')as f:json.dump(results,f,indent=2,default=str)
    return results


# ═══════════════════════════════════════════════════════════
# Exp-4: Multi-operator
# ═══════════════════════════════════════════════════════════

@tvm.script.ir_module
class Reduction128:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, b: T.handle) -> None:
        T.func_attr({'global_symbol': 'main'})
        A = T.match_buffer(a, (128, 128), 'float32')
        B = T.match_buffer(b, (128,), 'float32')
        for i, j in T.grid(128, 128):
            with T.sblock('reduction'):
                vi, vj = T.axis.remap('SS', [i, j])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vj]

@tvm.script.ir_module
class Elemwise128:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({'global_symbol': 'main'})
        A = T.match_buffer(a, (128, 128), 'float32')
        B = T.match_buffer(b, (128, 128), 'float32')
        C = T.match_buffer(c, (128, 128), 'float32')
        for i, j in T.grid(128, 128):
            with T.sblock('elemwise'):
                vi, vj = T.axis.remap('SS', [i, j])
                C[vi, vj] = A[vi, vj] + B[vi, vj]


def exp4_multi_operator():
    print('\n'+'='*60)
    print('Exp-4: Multi-operator Baseline Pass Rate')
    print('='*60)
    all_results={}

    for mod, name in [(Matmul128,'MatMul_128'),(Reduction128,'Reduction_128'),
                       (Elemwise128,'Elemwise_128')]:
        sg=PostOrderApply()
        ctx=TuneContext(mod=mod,target=TARGET,space_generator=sg,task_name=f'exp4_{name}')
        ctx.space_generator._initialize_with_tune_context(ctx)
        ds=ctx.space_generator.generate_design_space(mod)
        insts=[d.trace.insts for d in ds]

        postprocs=Postproc.create('cuda');skip={'RewriteTensorize','RewriteUnboundBlock'}
        pps=[pp for pp in postprocs if type(pp).__name__ not in skip]

        total=200;passed=0;failed=Counter()
        random.seed(42);t0=time.time()

        for _ in range(total):
            t=Trace(insts[random.randint(0,len(insts)-1)],{})
            try:
                sch=Schedule(mod,debug_mask='all')
                t.apply_to_schedule(sch,remove_postproc=False)
                sch.enter_postproc();ok=True;bad=None
                for pp in pps:
                    if not pp.apply(sch):ok=False;bad=type(pp).__name__;break
                if ok:passed+=1
                else:failed[bad]+=1
            except:failed['exc']+=1

        elapsed=time.time()-t0
        rate=passed/total*100
        print(f'  {name}: {passed}/{total} passed ({rate:.1f}%) in {elapsed:.0f}s')
        print(f'    Fails: {dict(failed.most_common())}')
        all_results[name]={'passed':passed,'total':total,'rate':rate,'fails':dict(failed)}

    with open(f'{OUTDIR}/exp4_multi_op.json','w')as f:json.dump(all_results,f,indent=2)
    return all_results


# ═══════════════════════════════════════════════════════════
# Main: Run all experiments
# ═══════════════════════════════════════════════════════════

if __name__=='__main__':
    os.makedirs(OUTDIR,exist_ok=True)
    t_start=time.time()

    # Phase 1: Quick analysis (seconds)
    print('\n'+'#'*70)
    print('# PHASE 1: Search Space Analysis')
    print('#'*70)
    exp1a_search_space()

    # Phase 2: Full comparison (hours)
    print('\n'+'#'*70)
    print('# PHASE 2: Three-mode Comparison (MatMul 128^3)')
    print('# Each mode: max 5000 attempts or 3 valid found')
    print('#'*70)
    exp2_comparison()

    # Phase 3: Multi-operator (minutes)
    print('\n'+'#'*70)
    print('# PHASE 3: Multi-operator Baseline')
    print('#'*70)
    exp4_multi_operator()

    # Summary
    total_elapsed=time.time()-t_start
    print(f'\n{"="*70}')
    print(f'ALL EXPERIMENTS COMPLETE ({total_elapsed/60:.0f} min)')
    print(f'Data saved to: {OUTDIR}/')
    print(f'{"="*70}')
