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
"""Run the pre-registered Draft-LSE, G2, G1, G3, and CS/H CUDA ablation arms."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "experiments" / "e1_thread_filter.py"
ANALYZER = REPO_ROOT / "experiments" / "analyze_second_point_v1_0.py"
DEFAULT_ARMS = (
    "B",
    "CSH",
    "D",
    "D_G2",
    "D_G1G2",
    "CSH_D_G1G2",
    "D_G1G2_G3_SHADOW",
    "D_G1G2_G3",
    "CSH_D_G1G2_G3",
)
ARM_SWITCHES = {
    "B": ("off", "off", "off", "off", "off"),
    "CSH": ("enforce", "enforce", "off", "off", "off"),
    "D": ("off", "off", "enforce", "off", "off"),
    "D_G2": ("off", "off", "enforce", "project", "off"),
    "D_G1G2": ("off", "off", "enforce", "directed", "off"),
    "CSH_D_G1G2": ("enforce", "enforce", "enforce", "directed", "off"),
    "D_G1G2_G3_SHADOW": ("off", "off", "enforce", "directed", "shadow"),
    "D_G1G2_G3": ("off", "off", "enforce", "directed", "enforce"),
    "CSH_D_G1G2_G3": ("enforce", "enforce", "enforce", "directed", "enforce"),
    "D_SHADOW": ("off", "off", "shadow", "off", "off"),
    "CSH_D": ("enforce", "enforce", "enforce", "off", "off"),
    "CSH_D_G2": ("enforce", "enforce", "enforce", "project", "off"),
    "CSH_D_SHADOW": ("enforce", "enforce", "shadow", "off", "off"),
}


def _arms(value: str) -> tuple[str, ...]:
    arms = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [arm for arm in arms if arm not in ARM_SWITCHES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown arms: {unknown}")
    if len(set(arms)) != len(arms):
        raise argparse.ArgumentTypeError("arm list contains duplicates")
    return arms


def _command(args: argparse.Namespace, arm: str) -> list[str]:
    hard, sample, draft_ga, ga_t2, g3 = ARM_SWITCHES[arm]
    return [
        str(args.python),
        str(DRIVER),
        "run",
        "--mode",
        hard,
        "--constraint-sample-mode",
        sample,
        "--hard-filter-position",
        "trace",
        "--soft-mode",
        "off",
        "--draft-mode",
        "off",
        "--draft-ga-mode",
        draft_ga,
        "--draft-ga-verify-ratio",
        str(args.draft_ga_verify_ratio),
        "--draft-ga-direction-trials",
        str(args.draft_ga_direction_trials),
        "--draft-ga-direction-eps",
        str(args.draft_ga_direction_eps),
        "--draft-ga-g3-mode",
        g3,
        "--draft-ga-anneal-mode",
        args.draft_ga_anneal_mode,
        "--draft-ga-anneal-t0",
        str(args.draft_ga_anneal_t0),
        "--draft-ga-anneal-decay",
        str(args.draft_ga_anneal_decay),
        "--ga-t2-mode",
        ga_t2,
        "--seed",
        str(args.seed),
        "--run-id",
        f"{args.run_label}_{arm}",
        "--work-root",
        str(args.work_root),
        "--temp-dir",
        str(args.temp_dir),
        "--cuda-home",
        str(args.cuda_home),
        "--trials",
        str(args.trials),
        "--trials-per-iter",
        str(args.trials_per_iter),
        "--population-size",
        str(args.population_size),
        "--init-min-unmeasured",
        str(args.init_min_unmeasured),
        "--exact-init-population",
        "--init-sample-batch-size",
        str(args.init_sample_batch_size),
        "--genetic-iters",
        str(args.genetic_iters),
        "--xgb-warmup-samples",
        str(args.xgb_warmup_samples),
        "--repetitions-note",
        args.repetitions_note,
    ]


def _environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    build_lib = REPO_ROOT / "build-prune" / "lib"
    local_python = REPO_ROOT / ".local" / "python"
    env["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (REPO_ROOT / "python", local_python)
    )
    env["TVM_LIBRARY_PATH"] = str(build_lib)
    env["PATH"] = os.pathsep.join(
        part
        for part in (str(build_lib), str(args.runtime_bin), env.get("PATH", ""))
        if part
    )
    return env


def _run_and_tee(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "second_point_corrected_draft_lse_g1_g2_g3",
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": args.seed,
        "arms": list(args.arms),
        "architecture_contract": (
            "Draft-only GA; zero learned-model calls inside GA; one top-K learned-model "
            "verification call after each search batch; G3 reward only from paired Runner "
            "latencies"
        ),
        "common_config": {
            "trials": args.trials,
            "trials_per_iter": args.trials_per_iter,
            "population_size": args.population_size,
            "init_min_unmeasured": args.init_min_unmeasured,
            "init_sample_batch_size": args.init_sample_batch_size,
            "genetic_iters": args.genetic_iters,
            "xgb_warmup_samples": args.xgb_warmup_samples,
            "draft_ga_verify_ratio": args.draft_ga_verify_ratio,
            "draft_ga_direction_trials": args.draft_ga_direction_trials,
            "draft_ga_direction_eps": args.draft_ga_direction_eps,
            "g3_window_size": 50,
            "g3_cold_start_measurements": 32,
            "g3_exploration_scale": 1.0,
            "draft_ga_anneal_mode": args.draft_ga_anneal_mode,
            "draft_ga_anneal_t0": args.draft_ga_anneal_t0,
            "draft_ga_anneal_decay": args.draft_ga_anneal_decay,
        },
        "commands": {
            arm: subprocess.list2cmdline(_command(args, arm)) for arm in args.arms
        },
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--arms", type=_arms, default=DEFAULT_ARMS)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--run-label", default="formal")
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv-tune" / "python.exe")
    parser.add_argument("--temp-dir", type=Path, default=Path("F:/tvm_tmp"))
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"),
    )
    parser.add_argument("--runtime-bin", type=Path, default=Path("D:/miniconda/Library/bin"))
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--trials-per-iter", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=128)
    parser.add_argument("--init-min-unmeasured", type=int, default=50)
    parser.add_argument("--init-sample-batch-size", type=int, default=64)
    parser.add_argument("--genetic-iters", type=int, default=4)
    parser.add_argument("--xgb-warmup-samples", type=int, default=0)
    parser.add_argument("--draft-ga-verify-ratio", type=float, default=0.25)
    parser.add_argument("--draft-ga-direction-trials", type=int, default=4)
    parser.add_argument("--draft-ga-direction-eps", type=float, default=0.10)
    parser.add_argument(
        "--draft-ga-anneal-mode", choices=("off", "shadow", "enforce"), default="off"
    )
    parser.add_argument("--draft-ga-anneal-t0", type=float, default=0.10)
    parser.add_argument("--draft-ga-anneal-decay", type=float, default=0.95)
    parser.add_argument("--repetitions-note", default="formal_single_seed")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch CUDA tuning; without this flag only print the immutable plan",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip tuning and validate existing results under --work-root",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.draft_ga_anneal_mode != "off":
        invalid_arms = [arm for arm in args.arms if ARM_SWITCHES[arm][2] != "enforce"]
        if invalid_arms:
            raise ValueError(
                "G4 annealing requires Draft-GA enforce arms; invalid arms: "
                f"{invalid_arms}"
            )
    args.work_root = args.work_root.resolve()
    plan = _plan(args)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute and not args.analyze_only:
        return

    if not args.analyze_only:
        args.work_root.mkdir(parents=True, exist_ok=True)
        if list(args.work_root.rglob("result.json")):
            raise FileExistsError(
                f"Refusing to mix a new run with existing results under {args.work_root}"
            )
        (args.work_root / "command_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        env = _environment(args)
        completed: list[str] = []
        for arm in args.arms:
            print(f"\n===== Running {arm} =====", flush=True)
            _run_and_tee(
                _command(args, arm), args.work_root / f"driver_{arm}.log", env
            )
            completed.append(arm)
            (args.work_root / "execution_state.json").write_text(
                json.dumps({"completed_arms": completed}, indent=2), encoding="utf-8"
            )

    analyze_command = [
        str(args.python),
        str(ANALYZER),
        "--root",
        str(args.work_root),
        "--required-arms",
        ",".join(args.arms),
    ]
    subprocess.run(analyze_command, cwd=REPO_ROOT, env=_environment(args), check=True)


if __name__ == "__main__":
    main()
