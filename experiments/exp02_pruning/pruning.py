"""
Bidirectional pruning model (Plan C, section 3.4).
Pure arithmetic checks — no TVM calls needed.
"""
import math


class GPU:
    """RTX 3050 Laptop (sm_86)"""
    shmem = 49152       # 48KB per block
    shmem_sm = 102400   # 100KB per SM
    max_threads = 1024
    threads_sm = 1536
    warp = 32


def estimate_shmem(bi, bj, bk, dtype_bytes=4):
    """Double-buffered matmul shared memory estimate."""
    read = (bi * bk + bk * bj) * dtype_bytes * 2
    write = bi * bj * dtype_bytes
    return read + write


def check_upward(bi, bj, bk, threads, gpu=GPU):
    """
    Upward pruning (hardware ceiling).
    Returns (ok, reason).
    """
    if bi is None:
        return False, 'no_tile_data'

    # 1. Thread count
    if threads is not None:
        if threads < gpu.warp or threads > gpu.max_threads:
            return False, 'threads'

    # 2. Shared memory
    if bk is not None and bj is not None:
        shmem = estimate_shmem(bi, bj, bk)
        if shmem > gpu.shmem:
            return False, 'shmem'

    return True, 'ok'


def check_downward(bi, bj, bk, threads, warp_tile=(32, 32, 16), gpu=GPU):
    """
    Downward pruning (performance floor).
    Assumes upward already passed. Returns (ok, reason).
    """
    wi, wj, wk = warp_tile

    # 1. Block >= warp tile
    if bi < wi or (bj and bj < wj) or (bk and bk < wk):
        return False, 'too_small'

    # 2. Block/warp divisibility (no tail warps)
    if bi % wi != 0:
        return False, 'bi_not_divisible'
    if bj and bj % wj != 0:
        return False, 'bj_not_divisible'
    if bk and bk % wk != 0:
        return False, 'bk_not_divisible'

    # 3. Data reuse (GEMM: >= 8 FLOP/byte)
    if bj is not None and bk is not None:
        flops = 2 * bi * bj * bk
        mem = (bi * bk + bk * bj) * 4
        if mem > 0 and flops / mem < 8.0:
            return False, 'low_reuse'

    return True, 'ok'


def extract_tile_params(trace):
    """
    Extract (bi, bj, bk, threads) from trace decisions.
    Returns None for values that can't be extracted.
    """
    dm = trace.decisions
    sp_factors = []  # n=5 (spatial)
    rd_factors = []  # n=3 (reduction)

    for key, val in list(dm.items()):
        ks = str(key)
        if 'sample_perfect_tile' not in ks.lower():
            continue
        try:
            vs = tuple(int(v.value) if hasattr(v, 'value') else int(v) for v in list(val))
        except:
            continue
        if 'n=5' in ks:
            sp_factors.append(vs)
        elif 'n=3' in ks:
            rd_factors.append(vs)

    if len(sp_factors) < 2:
        return None, None, None, None

    si = sp_factors[0]  # i-loop
    sj = sp_factors[1]  # j-loop
    bi, ti = si[0], si[2]  # block, thread factors
    bj, tj = sj[0], sj[2]

    bk = rd_factors[0][0] if rd_factors else None
    threads = ti * tj

    return bi, bj, bk, threads
