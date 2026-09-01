"""
Enumerate SamplePerfectTile space and compute legal rate.
"""
import sys, os
os.environ['USERPROFILE'] = 'C:\\tvm_home'
os.environ['HOME'] = 'C:\\tvm_home'

def enum_factors(extent, n_factors):
    """Enumerate all (f0,...,fn-1) where product = extent.
    Uses stars-and-bars: for 2^8 split 5 ways, C(12,4)=495.
    """
    from collections import Counter
    # Factorize
    x = extent
    primes = Counter()
    d = 2
    while d * d <= x:
        while x % d == 0:
            primes[d] += 1
            x //= d
        d += 1
    if x > 1:
        primes[x] += 1

    results = []
    prime_list = sorted(primes.keys(), reverse=True)

    def dfs(slot, current_factors, remaining):
        if slot == n_factors - 1:
            # Last slot gets all remaining
            last = 1
            for p, cnt in remaining.items():
                last *= p ** cnt
            results.append(tuple(current_factors + [last]))
            return

        # Distribute remaining primes to current slot
        if not remaining:
            results.append(tuple(current_factors + [1] * (n_factors - slot)))
            return

        p = prime_list[0]
        max_e = remaining.get(p, 0)

        for e in range(max_e + 1):
            new_rem = remaining.copy()
            if e > 0:
                new_rem[p] -= e
                if new_rem[p] == 0:
                    del new_rem[p]

            # Recurse for next prime or next slot
            if len(prime_list) > 1:
                def dfs_p(pi, val, rem):
                    if pi == len(prime_list):
                        dfs(slot + 1, current_factors + [val], rem)
                        return
                    pp = prime_list[pi]
                    max_ep = rem.get(pp, 0)
                    for ep in range(max_ep + 1):
                        new_r = rem.copy()
                        if ep > 0:
                            new_r[pp] -= ep
                            if new_r[pp] == 0: del new_r[pp]
                        dfs_p(pi + 1, val * (pp ** ep), new_r)
                dfs_p(1, p ** e, new_rem)
            else:
                dfs(slot + 1, current_factors + [p ** e], new_rem)

    dfs(0, [], primes)
    return results


# GPU spec
SHMEM = 49152      # 48KB
MAX_THREADS = 1024
MIN_THREADS = 32
WARP = 32
MAX_VTHREAD = 8

EXTENT = 128
print(f"Enumerating... MatMul({EXTENT},{EXTENT},{EXTENT})")
spatial = enum_factors(EXTENT, 5)   # i and j loops
reduction = enum_factors(EXTENT, 3) # k loop

print(f"i-loop (n=5): {len(spatial)}")
print(f"j-loop (n=5): {len(spatial)}")
print(f"k-loop (n=3): {len(reduction)}")
print(f"Total: {len(spatial)**2 * len(reduction):,}")

# Analyze: for each spatial factor combo, classify block tile
bi_dist = {}
for f in spatial:
    bi = f[0]
    bi_dist[bi] = bi_dist.get(bi, 0) + 1
print(f"\nbi distribution (1st factor of 5):")
for k, v in sorted(bi_dist.items()):
    print(f"  {k:3d}: {v:4d} combos")

# What makes a factor combo "legal"?
# Check per-factor constraints
def classify(f_i, f_j, f_k):
    """Classify a full factor assignment. Returns reason if illegal, or 'legal'."""
    bi, vi, ti, si1, si2 = f_i   # block, vthread, thread, serial, serial
    bj, vj, tj, sj1, sj2 = f_j
    bk, vk, tk = f_k             # block, serial, thread

    # 1. Warp alignment of block tile
    if bi < WARP or bi % WARP != 0:
        return 'warp_align'

    # 2. Thread count
    threads = ti * tj
    if threads < MIN_THREADS or threads > MAX_THREADS:
        return 'threads'

    # 3. Vthread limit
    vthreads = vi * vj
    if vthreads > MAX_VTHREAD:
        return 'vthread'

    # 4. Shared memory estimate
    shmem = (bi * bk + bk * bj) * 4 * 2 + bi * bj * 4
    if shmem > SHMEM:
        return 'shmem'

    return 'legal'


# Quick stats on a sample (full 11M is too large to enumerate all)
import random
random.seed(42)
n_sample = 100000
reasons = {}
legal_samples = []

# Weigh by actual distribution: each spatial/reduction combo is equally likely
all_spatial = list(spatial)
all_rd = list(reduction)

for _ in range(n_sample):
    f_i = random.choice(all_spatial)
    f_j = random.choice(all_spatial)
    f_k = random.choice(all_rd)
    reason = classify(f_i, f_j, f_k)
    reasons[reason] = reasons.get(reason, 0) + 1
    if reason == 'legal':
        legal_samples.append((f_i, f_j, f_k))

print(f"\n=== SAMPLED SPACE (n={n_sample:,}) ===")
for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
    pct = count / n_sample * 100
    print(f"  {reason:15s}: {count:6,d} ({pct:.1f}%)")

# What are the legal combos like?
print(f"\n=== LEGAL SAMPLES ({len(legal_samples)}) ===")
seen_bi = set()
for f_i, f_j, f_k in legal_samples[:10]:
    bi, bj, bk = f_i[0], f_j[0], f_k[0]
    threads = f_i[2] * f_j[2]
    shmem = (bi*bk + bk*bj)*8 + bi*bj*4
    seen_bi.add((bi, bj, bk))
    print(f"  bi={bi:3d} bj={bj:3d} bk={bk:3d} thr={threads:4d} shmem={shmem:6d}")

print(f"\n  Unique block tiles in legal samples: {len(seen_bi)}")
