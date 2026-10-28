"""Smoke test: torch 2.11 ResNet18 -> Relax via relax.frontend.torch."""
import os
import sys

sys.path.insert(0, "F:/tvm/python")
os.environ["TVM_LIBRARY_PATH"] = "F:/tvm/build-prune/lib"
os.environ["TEMP"] = "F:/tvm_tmp"
os.environ["TMP"] = "F:/tvm_tmp"

import torch
import torchvision.models as models

print("torch", torch.__version__, "cuda", torch.cuda.is_available())

model = models.resnet18(weights=None).eval()
x = torch.randn(1, 3, 224, 224)

from torch import export as torch_export

exported = torch_export.export(model, (x,))

from tvm.relax.frontend.torch import from_exported_program

mod = from_exported_program(exported)
print("Relax functions:", len(mod.functions))
print("OK: ResNet18 -> Relax succeeded")

# 下放 Relax 算子到 TIR
from tvm import relax

legalized = relax.transform.LegalizeOps()(mod)
from tvm import tirx

tir_funcs = []
for gv in legalized.get_global_vars():
    f = legalized[gv]
    if isinstance(f, tirx.PrimFunc):
        tir_funcs.append(gv.name_hint)
print("LegalizeOps 后 TIR PrimFunc 数:", len(tir_funcs))
for name in tir_funcs:
    print("  -", name)
