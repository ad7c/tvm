"""
Shared utilities for bidirectional pruning experiments.
MatMul workload, GPU spec, postproc helpers, analytical pruning model.
"""
import sys, os, json, math, time
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

sys.path.insert(0, 'F:/tvm/python')
os.environ['TVM_LIBRARY_PATH'] = 'F:/tvm/build/lib'


# ═══════════════════════════════════════════════════════
# GPU Spec — RTX 3050 Laptop (sm_86)
# ═══════════════════════════════════════════════════════

@dataclass
class GPUSpec:
    name: str = 'RTX 3050 Laptop (sm_86)'
    max_shmem_per_block: int = 49152      # 48KB
    max_shmem_per_sm: int = 102400        # 100KB
    max_threads_per_block: int = 1024
    max_threads_per_sm: int = 1536
    warp_size: int = 32


GPU = GPUSpec()


# ═══════════════════════════════════════════════════════
# MatMul TIR module
# ═══════════════════════════════════════════════════════

def make_matmul_module(m=256, n=256, k=256, dtype='float32'):
    import tvm
    from tvm.script import tirx as T

    if dtype == 'float32':
        @tvm.script.ir_module
        class Matmul:
            @T.prim_func(s_tir=True)
            def main(a: T.handle, b: T.handle, c: T.handle) -> None:
                T.func_attr({'global_symbol': 'main'})
                A = T.match_buffer(a, (m, k), 'float32')
                B = T.match_buffer(b, (k, n), 'float32')
                C = T.match_buffer(c, (m, n), 'float32')
                for i, j, kk in T.grid(m, n, k):
                    with T.sblock('matmul'):
                        vi, vj, vk = T.axis.remap('SSR', [i, j, kk])
                        with T.init():
                            C[vi, vj] = T.float32(0)
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        return Matmul
    raise NotImplementedError(f'dtype={dtype}')


def make_target():
    from tvm.target import Target
    return Target({
        'kind': 'cuda', 'arch': 'sm_86',
        'max_threads_per_block': GPU.max_threads_per_block,
        'max_shared_memory_per_block': GPU.max_shmem_per_block,
        'thread_warp_size': GPU.warp_size,
    })


# ═══════════════════════════════════════════════════════
# Postproc helpers
# ═══════════════════════════════════════════════════════

def init_tvm():
    from tvm_ffi import register_global_func
    from tvm.s_tir.meta_schedule.builder.local_builder import default_build, default_export
    register_global_func('default')(default_build)
    register_global_func('default_export')(default_export)


def get_postprocs(skip=('RewriteTensorize', 'RewriteUnboundBlock')):
    from tvm.s_tir.meta_schedule.postproc import Postproc
    all_pps = Postproc.create('cuda')
    return [pp for pp in all_pps if type(pp).__name__ not in skip], \
           [type(pp).__name__ for pp in all_pps if type(pp).__name__ not in skip]


def apply_postprocs(sch, pps):
    """Return (ok, first_failing_pp_name)."""
    sch.enter_postproc()
    for pp in pps:
        pp_name = type(pp).__name__
        try:
            if not pp.apply(sch):
                return False, pp_name
        except Exception as e:
            return False, f'{pp_name}({type(e).__name__})'
    return True, None


# ═══════════════════════════════════════════════════════
# Analytical pruning model
# ═══════════════════════════════════════════════════════

def get_divisors(n: int) -> List[int]:
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.add(i); divs.add(n // i)
    return sorted(divs)


def estimate_shmem(bi, bj, bk, dtype_bytes=4, double_buf=True, write_cache=True):
    read = (bi * bk + bk * bj) * dtype_bytes
    if double_buf: read *= 2
    write = bi * bj * dtype_bytes if write_cache else 0
    return read + write


def check_single_pruning(bi, bj, bk, threads, gpu=GPU):
    """Upward (hardware ceiling): shmem + threads + warp alignment."""
    if bi is None: return False
    if estimate_shmem(bi, bj, bk) > gpu.max_shmem_per_block: return False
    if threads and (threads < gpu.warp_size or threads > gpu.max_threads_per_block): return False
    if bi < gpu.warp_size or bi % gpu.warp_size != 0: return False
    return True


def check_double_pruning(bi, bj, bk, threads, warp_tile=(32, 32, 16), gpu=GPU):
    """Upward + downward (performance floor): divisibility + reuse."""
    if not check_single_pruning(bi, bj, bk, threads, gpu): return False
    wi, wj, wk = warp_tile
    if bi < wi or bj < wj or bk < wk: return False
    if bi % wi != 0 or bj % wj != 0 or bk % wk != 0: return False
    flops = 2 * bi * bj * bk
    mem = (bi * bk + bk * bj) * 4
    if mem > 0 and flops / mem < 8.0: return False
    return True


def enumerate_pruned_space(i_ext=256, j_ext=256, k_ext=256, mode='double'):
    """Enumerate divisor space and return [(bi,bj,bk,threads), ...] after pruning."""
    check = check_double_pruning if mode == 'double' else check_single_pruning
    results = []
    for bi in get_divisors(i_ext):
        ti = i_ext // bi
        for bj in get_divisors(j_ext):
            tj = j_ext // bj
            threads = ti * tj
            if threads < GPU.warp_size or threads > GPU.max_threads_per_block: continue
            for bk in get_divisors(k_ext):
                if check(bi, bj, bk, threads):
                    results.append((bi, bj, bk, threads))
    return results
