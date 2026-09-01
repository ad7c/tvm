"""Manually construct a valid GPU schedule for MatMul 64^3."""
import sys,os,numpy as np,time
os.environ['USERPROFILE']='C:\\tvm_home';os.environ['HOME']='C:\\tvm_home'
os.environ['TVM_LIBRARY_PATH']='F:/tvm/build/lib'
sys.path.insert(0,'F:/tvm/python')
import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.schedule import Schedule
from tvm.s_tir.meta_schedule.postproc import Postproc

@tvm.script.ir_module
class M:
    @T.prim_func(s_tir=True)
    def main(a:T.handle,b:T.handle,c:T.handle)->None:
        T.func_attr({'global_symbol':'main'})
        A=T.match_buffer(a,(64,64),'float32');B=T.match_buffer(b,(64,64),'float32');C=T.match_buffer(c,(64,64),'float32')
        for i,j,k in T.grid(64,64,64):
            with T.sblock('mm'):
                vi,vj,vk=T.axis.remap('SSR',[i,j,k])
                with T.init():C[vi,vj]=T.float32(0)
                C[vi,vj]=C[vi,vj]+A[vi,vk]*B[vk,vj]

tgt=Target({'kind':'cuda','arch':'sm_86','max_threads_per_block':1024,'max_num_threads':1024,'max_shared_memory_per_block':49152,'thread_warp_size':32})

# Manual schedule:
# i: split (8,8) → block, thread
# j: split (8,8) → block, thread
# k: split (16,4) → outer reduction, inner reduction
# Safe shmem: (8*16+16*8)*8 + 8*8*4 = 2048 + 256 = 2304 < 49152
sch=Schedule(M,debug_mask='all')
blk=sch.get_sblock('mm')
loops=sch.get_loops(blk)
print(f'Loops: {len(loops)}')

i_loop,j_loop,k_loop=loops[0],loops[1],loops[2]

# Use SamplePerfectTile for proper factor decomposition
# i: sample 2 factors (block, thread)
fi = sch.sample_perfect_tile(i_loop, 2, 64)
si = sch.split(i_loop, fi)  # returns List[LoopRV] - the new inner loops
fj = sch.sample_perfect_tile(j_loop, 2, 64)
sj = sch.split(j_loop, fj)
fk = sch.sample_perfect_tile(k_loop, 2, 64)
sk = sch.split(k_loop, fk)

print(f'Factors: i={[int(f) for f in fi]} j={[int(f) for f in fj]} k={[int(f) for f in fk]}')

# Bind spatial
sch.reorder([si[0], sj[0], sk[0], si[1], sj[1], sk[1]])
blk_fused = sch.fuse([si[0], sj[0]])
sch.bind(blk_fused, 'blockIdx.x')
thr_fused = sch.fuse([si[1], sj[1]])
sch.bind(thr_fused, 'threadIdx.x')

# Cache read
a_cache = sch.cache_read(blk, 0, 'shared')
b_cache = sch.cache_read(blk, 1, 'shared')
sch.compute_at(a_cache, sk[0])  # outer k loop
sch.compute_at(b_cache, sk[0])

# Postproc
postprocs=Postproc.create('cuda')
pps=[pp for pp in postprocs if type(pp).__name__ not in {'RewriteTensorize','RewriteUnboundBlock'}]
pp_names=[type(p).__name__ for p in pps]
print(f'Postproc: {pp_names}')

sch.enter_postproc()
ok=True
for pp in pps:
    pp_name=type(pp).__name__
    if not pp.apply(sch):
        print(f'  {pp_name}: REJECT')
        ok=False;break
    else:
        print(f'  {pp_name}: PASS')

if ok:
    print('\nAll postproc passed! Building...')
    rt=tvm.build(sch.mod,target=tgt)
    dev=tvm.cuda(0)
    a=np.random.randn(64,64).astype('float32')
    b=np.random.randn(64,64).astype('float32')
    c=np.zeros((64,64),'float32')
    a_g=tvm.runtime.ndarray.array(a,dev)
    b_g=tvm.runtime.ndarray.array(b,dev)
    c_g=tvm.runtime.ndarray.array(c,dev)
    te=rt.time_evaluator(rt.entry_name,dev,number=10,repeat=5)
    result=te(a_g,b_g,c_g)
    c_out=c_g.numpy()
    max_err=np.max(np.abs(c_out-a@b))
    print(f'  Latency: {result.mean*1000:.3f}ms')
    print(f'  GFLOPS: {2*64**3/(result.mean*1e9):.1f}')
    print(f'  Max error: {max_err:.2e} {"OK" if max_err<1e-3 else "WRONG"}')
else:
    print('\nPostproc failed')
