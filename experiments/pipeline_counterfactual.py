# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""CPU-only causal controls for the archived MatMul pipeline failures.

Observe the original RewriteReductionBlock, then perturb only eight previously
diagnosed failures. No production filtering rule, sampling, or GPU performance
claim. All postprocessors and the Verify lowering pipeline remain TVM originals.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

from diagnose_fixed_trace_pool import VerifyObserver, append_row
from fixed_trace_pool import (
    ROOT, ReplayExperiment, SOURCE_FILES, configure_environment, json_hash,
    plain_json, sha256, write_json,
)


PIPELINE_KEYS = {
    "software_pipeline_stage", "software_pipeline_order", "software_pipeline_async_stages",
}


def pipeline_summary(tvm, mod):
    """Pre-lowering structural observation, NOT an InjectSoftwarePipeline oracle."""
    pipelines = []

    def visit(node):
        if not isinstance(node, tvm.tirx.For) or "software_pipeline_stage" not in node.annotations:
            return
        body = node.body
        if isinstance(body, tvm.tirx.SBlockRealize):
            body = body.block.body
        components = list(body.seq) if isinstance(body, tvm.tirx.SeqStmt) else [body]
        blocks = []

        def collect(block):
            if isinstance(block, tvm.tirx.SBlock):
                blocks.append({"name": block.name_hint, "has_init": block.init is not None})

        tvm.tirx.stmt_functor.post_order_visit(node.body, collect)
        pipelines.append({
            "loop": node.loop_var.name, "extent": str(node.extent),
            "stage": [int(x) for x in node.annotations["software_pipeline_stage"]],
            "order": [int(x) for x in node.annotations["software_pipeline_order"]],
            "raw_direct_body_count": len(components), "blocks": blocks,
        })

    for func in mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return pipelines


def without_pipeline(tvm, trace):
    removed, kept = [], []
    for index, inst in enumerate(trace.insts):
        if inst.kind.name == "Annotate" and str(inst.attrs[0]) in PIPELINE_KEYS:
            if len(inst.outputs):
                raise ValueError("Cannot remove an annotation producing RVs")
            removed.append({"index": index, "key": str(inst.attrs[0])})
        else:
            kept.append(inst)
    if len(removed) != 3 or {row["key"] for row in removed} != PIPELINE_KEYS:
        raise ValueError("This control requires exactly one three-annotation pipeline")
    decisions = {inst: trace.get_decision(inst) for inst in kept
                 if trace.get_decision(inst) is not None}
    return tvm.s_tir.Trace(kept, decisions), {"removed_annotations": removed}


def outer_two(trace):
    # Workload-specific perturbation: this archived MatMul template has one
    # three-way reduction tile. This is NOT a generic reduction identification rule.
    matches = [(index, inst, [int(x) for x in trace.get_decision(inst)])
               for index, inst in enumerate(trace.insts)
               if inst.kind.name == "SamplePerfectTile" and len(inst.outputs) == 3]
    if len(matches) != 1 or matches[0][2][0] != 1:
        raise ValueError("Expected a unique three-way reduction tile with outer factor one")
    index, inst, old = matches[0]
    donor = next((i for i in range(1, len(old)) if old[i] % 2 == 0), None)
    if donor is None:
        raise ValueError("No even inner reduction factor")
    new = old.copy()
    new[0], new[donor] = 2, new[donor] // 2
    return trace.with_decision(inst, new, remove_postproc=True), {
        "instruction_index": index, "old_tile": old, "new_tile": new,
        "scope": "MatMul128 template-specific causal perturbation",
    }


def hoist_init(tvm, schedule):
    # Deliberately narrow pilot: a single unit pipeline contains one reduction
    # block initialized by a constant zero store, without BufferLoad dependencies.
    block_rv = schedule.get_sblock("m")
    block = schedule.get(block_rv)
    if not isinstance(block.init, tvm.tirx.BufferStore):
        raise ValueError("Pilot expects one neutral constant init store")
    value = block.init.value
    if not isinstance(value, (tvm.tirx.FloatImm, tvm.tirx.IntImm)) or float(value.value) != 0:
        raise ValueError("Pilot only hoists zero initialization")
    choices = [rv for rv in schedule.get_loops(block_rv)
               if "software_pipeline_stage" in schedule.get(rv).annotations]
    if len(choices) != 1 or int(schedule.get(choices[0]).extent) != 1:
        raise ValueError("Pilot expects one unit-extent annotated ancestor")
    loop = schedule.get(choices[0])
    schedule.decompose_reduction(block_rv, choices[0])
    return {"block": block.name_hint, "loop": loop.loop_var.name,
            "operation": "decompose_reduction before original RewriteReductionBlock"}


class ReductionObserver:
    def __init__(self, postproc, tvm):
        self.postproc, self.tvm = postproc, tvm
        self.reset(False)

    def reset(self, repair):
        self.repair = repair
        self.before = self.after_repair = self.after = None
        self.repair_detail = None

    def apply(self, schedule):
        self.before = schedule.mod
        if self.repair:
            self.repair_detail = hoist_init(self.tvm, schedule)
            self.after_repair = schedule.mod
        passed = self.postproc.apply(schedule)
        self.after = schedule.mod
        return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--library-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "command.json", {
        "argv": sys.argv, "utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
    })
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    old_identity = pool["identity"]
    bridge_identity = json.loads((args.bridge / "identity.json").read_text(encoding="utf-8"))
    bridge_summary = json.loads((args.bridge / "bridge_summary.json").read_text(encoding="utf-8"))
    if not bridge_summary["gate_passed"] or sha256(args.pool) != bridge_identity["pool_sha256"]:
        raise ValueError("Pool/bridge identity mismatch")
    bridge_path = args.bridge / "bridge.jsonl"
    if sha256(bridge_path) != bridge_summary["bridge_sha256"]:
        raise ValueError("Bridge events changed")
    old_rows = {row["candidate_id"]: row for row in
                (json.loads(line) for line in bridge_path.read_text(encoding="utf-8").splitlines())
                if row["arm"] == "baseline"}
    selected = {key for key, row in old_rows.items() if row["verify_reason"] == "lowering_exception"}
    if len(selected) != 8:
        raise ValueError("Expected the eight archived lowering failures")
    reference = Path(old_identity["reference_result"])
    if sha256(reference) != old_identity["reference_sha256"]:
        raise ValueError("Reference changed")
    current_source = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    if current_source != bridge_identity["current_source_files"]:
        raise ValueError("Source differs from validated diagnostic bridge")
    configure_environment(args.library_dir.resolve())
    experiment = ReplayExperiment(reference, pool["seed"],
                                  expected_library_sha256=args.library_sha256.lower())
    tvm = experiment.tvm
    if sha256(experiment.library) != bridge_identity["loaded_library_sha256"]:
        raise ValueError("DLL differs from validated bridge")
    if experiment.postproc_names != old_identity["postprocs"]:
        raise ValueError("Postprocessor chain changed")
    if json.loads(str(experiment.target)) != json.loads(old_identity["target"]):
        raise ValueError("Target changed")
    workload = args.pool.parent / "workload_tvm.json"
    if sha256(workload) != old_identity["workload_sha256"]:
        raise ValueError("Workload snapshot changed")
    tvm.ir.assert_structural_equal(experiment.mod, tvm.ir.load_json(workload.read_text(encoding="utf-8")))
    reduction_index = experiment.postproc_names.index("RewriteReductionBlock")
    reduction = ReductionObserver(experiment.postprocs[reduction_index], tvm)
    experiment.postprocs[reduction_index] = reduction
    verify_index = experiment.postproc_names.index("VerifyGPUCode")
    verify = VerifyObserver(experiment.postprocs[verify_index], tvm.get_global_func(
        "s_tir.meta_schedule.testing.GetLastVerifyGPUCodeFailureReason"), tvm)
    experiment.postprocs[verify_index] = verify
    write_json(args.output / "identity.json", {
        "pool_sha256": sha256(args.pool), "bridge_sha256": sha256(bridge_path),
        "loaded_library": str(experiment.library), "loaded_library_sha256": sha256(experiment.library),
        "current_source_files": current_source, "postprocs": experiment.postproc_names,
        "target": str(experiment.target), "selected_ids": sorted(selected),
        "driver_sha256": sha256(__file__),
        "scope": "CPU diagnostic; no GPU correctness, timings include observers, not speedup data",
    })
    for name in (Path(__file__).name, "fixed_trace_pool.py", "diagnose_fixed_trace_pool.py"):
        shutil.copy2(Path(__file__).with_name(name), args.output / ("snapshot_" + name))
    candidates = [row for row in pool["candidates"] if row["generation_status"] == "complete"]
    rows = []
    with (args.output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for variant in ("original", "no_pipeline", "outer2", "hoist_init"):
            subset = candidates if variant == "original" else [c for c in candidates if c["id"] in selected]
            for index, candidate in enumerate(subset):
                key = candidate["id"]
                trace = experiment.decode_trace(candidate["trace_tvm_json"])
                if json_hash(plain_json(trace.as_json(True))) != candidate["trace_sha256"]:
                    raise ValueError("Trace identity mismatch")
                detail = None
                if variant == "no_pipeline":
                    trace, detail = without_pipeline(tvm, trace)
                elif variant == "outer2":
                    trace, detail = outer_two(trace)
                reduction.reset(variant == "hoist_init")
                verify.reset()
                row, schedule = experiment.evaluate(trace, candidate["seed"], "baseline")
                row.update(candidate_id=key, variant=variant, verify_reason=verify.reason,
                           verify_called=verify.called, change=detail or reduction.repair_detail,
                           original_trace_sha256=candidate["trace_sha256"],
                           input_trace_sha256=json_hash(plain_json(trace.as_json(True))))
                row["structures"] = {}
                for phase, mod in (("before_reduction", reduction.before),
                                   ("after_repair", reduction.after_repair),
                                   ("after_reduction", reduction.after), ("verify_input", verify.input_mod)):
                    if mod is None:
                        continue
                    row["structures"][phase] = pipeline_summary(tvm, mod)
                    if key in selected:
                        prefix = args.output / f"{key}.{variant}.{phase}"
                        Path(str(prefix) + ".json").write_text(tvm.ir.save_json(mod), encoding="utf-8")
                        Path(str(prefix) + ".tir.txt").write_text(mod.script(), encoding="utf-8")
                if key in selected:
                    write_json(args.output / f"{key}.{variant}.trace.json", experiment.encode_trace(trace))
                    if schedule is not None:
                        (args.output / f"{key}.{variant}.accepted_mod.json").write_text(
                            tvm.ir.save_json(schedule.mod), encoding="utf-8")
                        (args.output / f"{key}.{variant}.postproc_trace.json").write_text(
                            tvm.ir.save_json(schedule.trace), encoding="utf-8")
                if variant == "original":
                    row["status_agrees"] = row["status"] == old_rows[key]["status"]
                    row["reason_agrees"] = row["verify_reason"] == old_rows[key]["verify_reason"]
                    if not row["status_agrees"] or not row["reason_agrees"]:
                        raise AssertionError(f"Original drift: {key}")
                    if row["accepted"]:
                        tvm.ir.assert_structural_equal(schedule.mod, tvm.ir.load_json(candidate["oracle_mod_tvm_json"]))
                        row["oracle_ir_equal"] = True
                append_row(stream, row)
                rows.append(row)
                if variant != "original" or (index + 1) % 32 == 0 or index + 1 == len(subset):
                    print(f"{variant} {index + 1}/{len(subset)}: {key} {row['status']} ({verify.reason})", flush=True)
    write_json(args.output / "summary.json", {
        "completed": True, "original_gate_passed": True,
        "original_candidates": len(candidates), "selected_failures": len(selected),
        "original_ir_checks": sum(row.get("oracle_ir_equal", False) for row in rows),
        "variants": {variant: {"evaluations": sum(row["variant"] == variant for row in rows),
                               "reason_counts": dict(Counter(row["verify_reason"] for row in rows if row["variant"] == variant)),
                               "status_counts": dict(Counter(row["status"] for row in rows if row["variant"] == variant))}
                     for variant in ("original", "no_pipeline", "outer2", "hoist_init")},
        "results_sha256": sha256(args.output / "results.jsonl"),
        "gpu_correctness": "not tested", "performance": "not measured",
    })


if __name__ == "__main__":
    main()
