"""Smoke: ResNet18 -> 32 TIR ops -> tune_tir (tiny budget), verify path works."""
import os
import sys

sys.path.insert(0, "F:/tvm/python")
os.environ["PYTHONPATH"] = "F:/tvm/python" + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
os.environ["TVM_LIBRARY_PATH"] = "F:/tvm/build-prune/lib"
os.environ["TEMP"] = "F:/tvm_tmp"
os.environ["TMP"] = "F:/tvm_tmp"

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

# 1. ResNet18 -> Relax -> LegalizeOps -> 32 TIR PrimFuncs
model = models.resnet18(weights=None).eval()
x = torch.randn(1, 3, 224, 224)
exported = torch_export.export(model, (x,))
relax_mod = from_exported_program(exported)
legalized = relax.transform.LegalizeOps()(relax_mod)

# 2. 打包成只含 TIR PrimFunc 的 IRModule（tune_tir 会自动挑 PrimFunc）
target = Target({
    "kind": "cuda",
    "arch": "sm_86",
    "max_threads_per_block": 1024,
    "max_num_threads": 1024,
    "max_shared_memory_per_block": 49152,
    "thread_warp_size": 32,
})

builder = LocalBuilder(max_workers=1, timeout_sec=60)
runner = LocalRunner(timeout_sec=60, evaluator_config=EvaluatorConfig(number=3, repeat=1, min_repeat_ms=100))
cost_model = XGBModel(num_warmup_samples=100, num_tuning_cores=1, adaptive_training=False)
strategy = EvolutionarySearch(population_size=128, init_measured_ratio=0.2, init_min_unmeasured=50,
                              max_fail_count=5, genetic_num_iters=4, genetic_mutate_prob=0.85,
                              genetic_max_fail_count=10, eps_greedy=0.10)

import time
t0 = time.perf_counter()
database = tir_integration.tune_tir(
    mod=legalized,
    target=target,
    work_dir="F:/tvm_tmp/e2e_smoke",
    max_trials_global=8,
    num_trials_per_iter=8,
    builder=builder,
    runner=runner,
    cost_model=cost_model,
    strategy=strategy,
    space="post-order-apply",
    num_tuning_cores=1,
    seed=20260917,
)
t1 = time.perf_counter()
print(f"=== tune 完成，耗时 {t1 - t0:.1f}s ===")
records = list(database.get_all_tuning_records())
finite = [r for r in records if r.run_secs and all(0 < s < 1e9 for s in r.run_secs)]
print(f"总 tuning records: {len(records)}, 有限延迟: {len(finite)}")
