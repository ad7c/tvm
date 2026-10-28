"""Plot per-operator best-so-far convergence for the slowest (bottleneck) operators."""
import sys, os, hashlib
sys.path.insert(0, "F:/tvm/python")
os.environ["TVM_LIBRARY_PATH"] = "F:/tvm/build-prune/lib"
os.environ["TEMP"] = "F:/tvm_tmp"; os.environ["TMP"] = "F:/tvm_tmp"

from tvm.s_tir.meta_schedule.database import JSONDatabase

WORK = "F:/tvm_tmp/e2e_resnet/B_seed20260921_20260921_164240"
db = JSONDatabase(
    path_workload=WORK + "/database_workload.json",
    path_tuning_record=WORK + "/database_tuning_record.json",
)
records = list(db.get_all_tuning_records())

def op_key(r):
    return hashlib.sha1(str(r.workload.mod).encode("utf-8")).hexdigest()[:12]

# 每个算子的所有 (trial_index, latency)
ops = {}
for i, r in enumerate(records):
    if r.run_secs and all(0 < s < 1e9 for s in r.run_secs):
        ops.setdefault(op_key(r), []).append((i, min(float(s) for s in r.run_secs) * 1e6))

# 每个算子的最终 best，找最慢的 3 个（瓶颈算子）
final_best = {k: min(v for _, v in lst) for k, lst in ops.items()}
slowest = sorted(final_best.items(), key=lambda kv: kv[1], reverse=True)[:3]

for k, fb in slowest:
    lst = ops[k]
    best_so_far = []
    b = float("inf")
    for idx, lat in lst:
        b = min(b, lat)
        best_so_far.append((idx, b))
    # 降采样
    n = len(best_so_far)
    step = max(1, n // 40)
    pts = best_so_far[::step]
    if pts[-1] != best_so_far[-1]:
        pts.append(best_so_far[-1])
    ymin, ymax = min(p[1] for p in pts), max(p[1] for p in pts)
    width = 55
    print(f"\n算子 {k}  最终 best={fb:.1f}us  ({n} 次测量):")
    for p in pts:
        bar = int((p[1] - ymin) / (ymax - ymin) * width) if ymax > ymin else 0
        print(f"{p[0]:5d} {p[1]:8.1f} |{'#' * bar}")
