"""Run tune_tir with PYTHONPATH set for popen_worker."""
import sys,os,time
os.environ['PYTHONPATH']='F:/tvm/python'
os.environ['USERPROFILE']='C:\\tvm_home'
os.environ['HOME']='C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
sys.path.insert(0,'F:/tvm/python')
from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build,default_export
register_global_func('default')(default_build)
register_global_func('default_export')(default_export)
import tvm,numpy as np
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

t0=time.time()
database=tir_integration.tune_tir(
    mod=M,target=tgt,work_dir='F:/tvm/experiments/data/tune_demo2',
    max_trials_global=8,num_trials_per_iter=4,
    strategy='evolutionary',space='post-order-apply',num_tuning_cores=1)
records=database.get_all_tuning_records()
print(f'\nDONE: {len(records)} records in {time.time()-t0:.0f}s')

if records:
    print('Records found! Meta Schedule CAN produce valid schedules.')
    for i,rec in enumerate(records[:3]):
        try:
            from tvm.s_tir.schedule import Schedule
            from tvm.s_tir.schedule._ffi_api import TraceApplyJSONToSchedule
            sch=Schedule(M,debug_mask='all')
            # TuningRecord has .trace attribute
            trace_json = rec.trace.as_json()
            TraceApplyJSONToSchedule(trace_json, sch)
            # Apply postproc
            from tvm.s_tir.meta_schedule.postproc import Postproc
            pps=[pp for pp in Postproc.create('cuda') if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]
            sch.enter_postproc()
            for pp in pps:
                if not pp.apply(sch):
                    raise RuntimeError(f'Postproc {type(pp).__name__} failed on replay')
            rt=tvm.build(sch.mod,target=tgt)
            dev=tvm.cuda(0)
            a=np.random.randn(128,128).astype('float32')
            b=np.random.randn(128,128).astype('float32')
            c=np.zeros((128,128),'float32')
            a_g=tvm.runtime.ndarray.array(a,dev)
            b_g=tvm.runtime.ndarray.array(b,dev)
            c_g=tvm.runtime.ndarray.array(c,dev)
            te=rt.time_evaluator(rt.entry_name,dev,number=10,repeat=3)
            r=te(a_g,b_g,c_g)
            err=np.max(np.abs(c_g.numpy()-a@b))
            print(f'  Rec{i}: {r.mean*1000:.3f}ms, {2*128**3/(r.mean*1e9):.0f}GFLOPS, err={err:.2e}')
        except Exception as e:
            print(f'  Rec{i}: {type(e).__name__}: {str(e)[:120]}')
else:
    print('No records - popen_worker likely failed')
