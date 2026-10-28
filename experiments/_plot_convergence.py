"""Plot convergence curve (end-to-end best-so-far latency vs trial) for one arm."""
import sys, os
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

# 按算子（workload 结构）分组，跟踪 best-so-far
import hashlib

def op_key(r):
    return hashlib.sha1(str(r.workload.mod).encode("utf-8")).hexdigest()[:12]

best = {}          # op_key -> best latency (s)
curve = []         # (trial_index, sum_of_best_so_far_us)
for i, r in enumerate(records):
    if r.run_secs and all(0 < s < 1e9 for s in r.run_secs):
        lat = min(float(s) for s in r.run_secs)
        k = op_key(r)
        if k not in best or lat < best[k]:
            best[k] = lat
    curve.append((i, sum(best.values()) * 1e6))

print(f"records={len(records)}  distinct_ops={len(best)}  final_sum={curve[-1][1]:.2f}us")

# 落 CSV
import csv
with open("F:/tvm_tmp/e2e_convergence_B.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial", "end_to_end_best_so_far_us"])
    w.writerows((c[0], round(c[1], 2)) for c in curve)

# 纯文本 ASCII 曲线（降采样到 ~60 点，x=trial, y=端到端 best-so-far us）
n = len(curve)
step = max(1, n // 60)
pts = curve[::step]
if pts[-1] != curve[-1]:
    pts.append(curve[-1])
ymin, ymax = min(p[1] for p in pts), max(p[1] for p in pts)
width = 60
print(f"\n端到端 best-so-far 收敛曲线 (x=trial 0..{n}, y={ymin:.0f}..{ymax:.0f} us):")
for p in pts:
    bar = int((p[1] - ymin) / (ymax - ymin) * width) if ymax > ymin else 0
    print(f"{p[0]:5d} {p[1]:8.1f} |{'#' * bar}")
print("saved csv: F:/tvm_tmp/e2e_convergence_B.csv")
