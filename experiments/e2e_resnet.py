"""End-to-end ResNet18 tuning experiment (independent).

This script is self-contained and does NOT touch other experiments:
  - its own work_dir under F:/tvm_tmp/e2e_resnet/
  - does not import or modify e1_thread_filter.py
  - writes result.json into the work_dir only

Arms:
  B    : all switches off (baseline)
  CSH  : constraint sampling + H-trace hard filter
  D    : draft-guided GA only
  D_G2 : draft-guided GA + G2 projection
  Full : CSH + draft-guided GA + G1v2/G2

Usage:
  python e2e_resnet.py --arm B --trials 1000 --seed 20260921
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "F:/tvm/python")

MODE_TO_VALUE = {"off": 0, "shadow": 1, "enforce": 2}
GA_T2_MODE_TO_VALUE = {"off": 0, "shadow": 1, "project": 2, "directed": 3}


def arm_env(arm: str) -> dict:
    """Map an arm name to its env-var switch values (mirrors e1_thread_filter.py)."""
    base = {
        "TVM_PRUNE_MODE": "0",  # hard thread filter (off)
        "TVM_THREAD_HARD_FILTER_MODE": "off",
        "TVM_THREAD_HARD_FILTER_POSITION": "trace",
        "TVM_THREAD_CONSTRAINT_SAMPLE_MODE": "off",
        "TVM_SOFT_FILTER_MODE": "0",
        "TVM_DRAFT_MODE": "0",
        "TVM_DRAFT_GA_MODE": "0",
        "TVM_GA_T2_MODE": "0",
    }
    if arm == "B":
        return base
    if arm == "CSH":
        base["TVM_PRUNE_MODE"] = "2"
        base["TVM_THREAD_HARD_FILTER_MODE"] = "enforce"
        base["TVM_THREAD_CONSTRAINT_SAMPLE_MODE"] = "enforce"
        return base
    if arm == "D":
        base["TVM_DRAFT_GA_MODE"] = "2"
        return base
    if arm == "D_G2":
        base["TVM_DRAFT_GA_MODE"] = "2"
        base["TVM_GA_T2_MODE"] = "2"  # project
        return base
    if arm == "Full":
        base["TVM_PRUNE_MODE"] = "2"
        base["TVM_THREAD_HARD_FILTER_MODE"] = "enforce"
        base["TVM_THREAD_CONSTRAINT_SAMPLE_MODE"] = "enforce"
        base["TVM_DRAFT_GA_MODE"] = "2"
        base["TVM_GA_T2_MODE"] = "3"  # directed (G1v2 + G2)
        return base
    raise ValueError(f"unknown arm: {arm}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["B", "CSH", "D", "D_G2", "Full"], required=True)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--trials-per-iter", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--population-size", type=int, default=64)
    args = parser.parse_args()

    # --- 1. 独立 work_dir（时间戳，避免混旧数据）---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = Path("F:/tvm_tmp/e2e_resnet") / f"{args.arm}_seed{args.seed}_{stamp}"
    temp_dir = Path("F:/tvm_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. 设环境变量（必须在 import tvm 之前）---
    for key, val in arm_env(args.arm).items():
        os.environ[key] = val
    os.environ["TVM_DRAFT_PROFILE"] = "rtx3050ti_sm86_explicit_v1"
    os.environ["TVM_DRAFT_PEAK_GFLOPS"] = "10752.0"
    os.environ["TVM_DRAFT_MEMORY_GBPS"] = "192.0"
    os.environ["TVM_DRAFT_SM_COUNT"] = "20"
    os.environ["TVM_DRAFT_ACTIVE_BLOCKS_PER_SM"] = "1"
    os.environ["TVM_DRAFT_SM_PARTITIONS"] = "4"
    os.environ["TVM_DRAFT_REG_CAP"] = "128.0"
    os.environ["TVM_LIBRARY_PATH"] = "F:/tvm/build-prune/lib"
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["PYTHONPATH"] = "F:/tvm/python" + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )

    # --- 3. import tvm（在 env 设置之后）---
    import torch
    import torchvision.models as models
    from torch import export as torch_export
    from tvm import relax, tirx
    from tvm.relax.frontend.torch import from_exported_program
    from tvm.target import Target
    from tvm.s_tir.meta_schedule import tir_integration
    from tvm.s_tir.meta_schedule.builder import LocalBuilder
    from tvm.s_tir.meta_schedule.runner import LocalRunner, EvaluatorConfig
    from tvm.s_tir.meta_schedule.cost_model import XGBModel
    from tvm.s_tir.meta_schedule.search_strategy import EvolutionarySearch

    # --- 4. ResNet18 -> Relax -> 32 TIR ops ---
    t0 = time.perf_counter()
    model = models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)
    exported = torch_export.export(model, (x,))
    relax_mod = from_exported_program(exported)
    legalized = relax.transform.LegalizeOps()(relax_mod)
    n_ops = sum(1 for gv in legalized.get_global_vars() if isinstance(legalized[gv], tirx.PrimFunc))
    setup_s = time.perf_counter() - t0

    target = Target({
        "kind": "cuda", "arch": "sm_86",
        "max_threads_per_block": 1024, "max_num_threads": 1024,
        "max_shared_memory_per_block": 49152, "thread_warp_size": 32,
    })

    builder = LocalBuilder(max_workers=1, timeout_sec=60)
    runner = LocalRunner(timeout_sec=60, evaluator_config=EvaluatorConfig(number=3, repeat=1, min_repeat_ms=100))
    cost_model = XGBModel(num_warmup_samples=100, num_tuning_cores=1, adaptive_training=False)
    strategy = EvolutionarySearch(
        population_size=args.population_size, init_measured_ratio=0.2, init_min_unmeasured=32,
        max_fail_count=5, genetic_num_iters=4, genetic_mutate_prob=0.85,
        genetic_max_fail_count=10, eps_greedy=0.10,
    )

    # --- 5. tune ---
    t1 = time.perf_counter()
    database = tir_integration.tune_tir(
        mod=legalized, target=target, work_dir=str(work_dir),
        max_trials_global=args.trials, num_trials_per_iter=args.trials_per_iter,
        builder=builder, runner=runner, cost_model=cost_model, strategy=strategy,
        space="post-order-apply", num_tuning_cores=1, seed=args.seed,
    )
    tune_s = time.perf_counter() - t1

    # --- 6. 记录结果 ---
    import re

    records = list(database.get_all_tuning_records())
    n_finite = sum(1 for r in records if r.run_secs and all(0 < s < 1e9 for s in r.run_secs))

    # 从 task_scheduler 日志解析"Total latency（端到端延迟代理）"和每算子表
    sched_log = work_dir / "logs" / "tvm.s_tir.meta_schedule.logging.task_scheduler.log"
    sched_text = sched_log.read_text(encoding="utf-8", errors="replace") if sched_log.exists() else ""

    total_latency_us = None
    per_op = {}
    for line in sched_text.splitlines():
        # Total latency：取最后一个（最终汇总）
        m = re.search(r"Total latency \(us\):\s*([\d.]+)", line)
        if m:
            total_latency_us = float(m.group(1))
        # 表格行：split by |，cols[1]=Name, cols[5]=Latency(us)
        if "|" not in line or line.strip().startswith("-"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 6:
            continue
        name, lat = cols[1], cols[5]
        if name in ("ID", "Name", "") or lat in ("N/A", "", "Latency"):
            continue
        try:
            per_op[name] = float(lat)
        except ValueError:
            pass

    result = {
        "arm": args.arm,
        "seed": args.seed,
        "trials": args.trials,
        "n_ops": n_ops,
        "setup_s": round(setup_s, 1),
        "tune_s": round(tune_s, 1),
        "n_records": len(records),
        "n_finite": n_finite,
        "total_latency_us": total_latency_us,
        "per_op_latency_us": per_op,
        "work_dir": str(work_dir),
    }

    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{args.arm}] ops={n_ops} setup={setup_s:.0f}s tune={tune_s:.0f}s "
          f"records={len(records)} finite={result['n_finite']} total_latency={result['total_latency_us']}us")
    print(f"result -> {out}")


if __name__ == "__main__":
    main()
