"""E1 测量协议：无剪枝 baseline 跑 N 次，记录最优延迟 + 墙钟 + 搜索指标，报均值±方差。

环境要求（已修复）：TVM_LIBRARY_PATH 指向 build-prune/lib，TEMP/TMP 必须 ASCII。
"""
import os
import shutil
import time

os.environ["TVM_LIBRARY_PATH"] = "F:/tvm/build-prune/lib"
os.environ["USERPROFILE"] = "C:\\tvm_home"
os.environ["HOME"] = "C:\\tvm_home"
os.environ["TEMP"] = "F:\\tvm_tmp"
os.environ["TMP"] = "F:\\tvm_tmp"
os.environ["CUDA_HOME"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
os.environ["CUDA_PATH"] = os.environ["CUDA_HOME"]
os.environ["PYTHONPATH"] = "F:/tvm/python"
import sys
sys.path.insert(0, "F:/tvm/python")

from tvm_ffi import register_global_func
from tvm.s_tir.meta_schedule.builder.local_builder import default_build, default_export

register_global_func("default")(default_build)
register_global_func("default_export")(default_export)

import re
import numpy as np
import tvm
from tvm.script import tirx as T
from tvm.target import Target
from tvm.s_tir.meta_schedule import tir_integration


@tvm.script.ir_module
class M128:
    @T.prim_func(s_tir=True)
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (128, 128), "float32")
        B = T.match_buffer(b, (128, 128), "float32")
        C = T.match_buffer(c, (128, 128), "float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("m"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


TGT = Target({
    "kind": "cuda", "arch": "sm_86",
    "max_threads_per_block": 1024, "max_num_threads": 1024,
    "max_shared_memory_per_block": 49152, "thread_warp_size": 32,
})


def run_once(tag):
    work_dir = f"F:/tvm_tmp/e1_baseline_{tag}"
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()
    db = tir_integration.tune_tir(
        mod=M128, target=TGT, work_dir=work_dir,
        max_trials_global=8, num_trials_per_iter=8,
        strategy="evolutionary", space="post-order-apply", num_tuning_cores=1,
    )
    wall = time.time() - t0
    recs = db.get_all_tuning_records()
    lat = [float(r.run_secs[0]) for r in recs if float(r.run_secs[0]) < 1e9]
    best = min(lat) if lat else float("inf")
    # 读日志拿 VerifyGPUCode 失败数（搜索阶段指标）
    log = f"{work_dir}/logs/tvm.s_tir.meta_schedule.logging.task_0_main.log"
    vgpu = 0
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        vgpu = sum(int(x) for x in re.findall(r"VerifyGPUCode\]: (\d+) failure", txt))
    return {"best_us": best * 1e6, "wall_s": wall, "n_measured": len(lat), "vgpu_fail": vgpu}


N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
print(f"=== E1 baseline：无剪枝，跑 {N} 次（max_trials=8）===")
results = []
for i in range(N):
    r = run_once(f"run{i}")
    results.append(r)
    print(f"  run{i}: best={r['best_us']:.2f} us  wall={r['wall_s']:.0f}s  "
          f"measured={r['n_measured']}  vgpu_fail={r['vgpu_fail']}", flush=True)

best_arr = np.array([r["best_us"] for r in results])
wall_arr = np.array([r["wall_s"] for r in results])
print(f"\n=== 汇总 ===")
print(f"  最优延迟: {best_arr.mean():.2f} ± {best_arr.std():.2f} us  (n={N})")
print(f"  墙钟:     {wall_arr.mean():.0f} ± {wall_arr.std():.0f} s")
