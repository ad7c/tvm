"""Pruner 式硬件启发式的早期原型：从 split 因子估算 shmem + threads，零 lowering。

核心公式（Pruner S3-L1MemAlloc 的 SSSRRSRS 适配版）：
  i-loop 5 因子 [I0,I1,I2,I3,I4]，j-loop [J0..J4]，k-loop 3 因子 [K0,K1,K2]
  threads  = I2 * J2                      （threadIdx.x = Fuse(I2,J2)）
  A_shmem  = (I1*I2*I3*I4) * (K1*K2)      （A 的 block tile × 内层 reduction）
  B_shmem  = (J1*J2*J3*J4) * (K1*K2)
  shmem    = (A_shmem + B_shmem) * dtype_bytes

适用范围仅是当前 MatMul-128、SSSRRSRS trace 形态。脚本读取的是已经通过
VerifyGPUCode 的 tuning records，只能观察启发式估计在这些合法样本上的表现；
它没有 replay/lowering，也没有读取 CompactBufferAllocation 的真实 shmem，
因此不能证明公式精确、过滤有效或具有通用的“零误杀”性质。
"""
import os

os.environ["TVM_LIBRARY_PATH"] = "F:/tvm/build-prune/lib"
os.environ["USERPROFILE"] = "C:\\tvm_home"
os.environ["HOME"] = "C:\\tvm_home"
os.environ["TEMP"] = "F:\\tvm_tmp"
os.environ["TMP"] = "F:\\tvm_tmp"
os.environ["CUDA_HOME"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
os.environ["PYTHONPATH"] = "F:/tvm/python"
import sys
sys.path.insert(0, "F:/tvm/python")

import json
import re


def load_records(path):
    recs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def extract_factors(record):
    """从 trace 决策提取 i/j/k 三个循环的 split 因子。"""
    insts = record[1][0][0]
    decisions = record[1][0][1]
    inst_by_idx = {i: inst for i, inst in enumerate(insts)}
    spt5 = []   # n=5 的（按 trace 顺序：先 i 后 j）
    spt3 = None  # n=3 的（k）
    for d in decisions:
        idx, val = d[0], d[1]
        inst = inst_by_idx[idx]
        if inst[0] == "SamplePerfectTile":
            n = inst[2][0]
            if n == 5:
                spt5.append(val)
            elif n == 3:
                spt3 = val
    I, J = spt5[0], spt5[1]
    K = spt3
    return I, J, K


def hardware_formula(I, J, K, dtype_bytes=4):
    """返回当前 trace 形态下的启发式 (shmem_bytes, threads) 估计。"""
    threads = I[2] * J[2]  # threadIdx.x = I2*J2
    A_shmem = (I[1] * I[2] * I[3] * I[4]) * (K[1] * K[2])
    B_shmem = (J[1] * J[2] * J[3] * J[4]) * (K[1] * K[2])
    shmem = (A_shmem + B_shmem) * dtype_bytes
    return shmem, threads


if __name__ == "__main__":
    path = "F:/tvm_tmp/e1_baseline_run0/database_tuning_record.json"
    recs = load_records(path)
    print(f"=== 在 {len(recs)} 个已通过 VerifyGPUCode 的候选上观察启发式估计 ===")
    print(f"{'idx':>3} {'I0..I4':<18} {'J0..J4':<18} {'K0..K2':<12} {'est_shmem':>9} {'est_thr':>7} {'≤48KB?':>7} {'≤1024?':>7}")
    all_ok = True
    for i, r in enumerate(recs):
        I, J, K = extract_factors(r)
        shmem, threads = hardware_formula(I, J, K)
        ok_sh = shmem <= 49152
        ok_th = threads <= 1024
        if not (ok_sh and ok_th):
            all_ok = False
        print(f"{i:>3} {str(I):<18} {str(J):<18} {str(K):<12} {shmem:>9} {threads:>7} {str(ok_sh):>7} {str(ok_th):>7}")

    print()
    if all_ok:
        print(
            ">>> 观察结果：这些已通过 VerifyGPUCode 的样本均未被启发式估计为超限；"
            "这不证明公式精确或具有通用零误杀性质。"
        )
    else:
        print(
            ">>> 观察结果：至少一个已通过 VerifyGPUCode 的样本被启发式估计为超限；"
            "当前原型不能作为硬过滤规则。"
        )
