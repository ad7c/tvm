# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

"""Export manuscript tables from the audited single-seed study summary."""

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "papers/figures/single_seed_20261010/summary.json"
OUTPUT = ROOT / "papers/tables"
GROUPS = (
    ("matmul128", 1, "MatMul128 / 1"),
    ("matmul128", 4, "MatMul128 / 4"),
    ("matmul256", 1, "MatMul256 / 1"),
    ("conv2d_nchw3x3", 1, "Conv / 1"),
    ("reduction4096", 1, "Reduction / 1"),
)
ARMS = ("B", "C", "D")


def row(cells):
    return " & ".join(str(cell) for cell in cells) + r" \\"


def table(caption, label, columns, headers, rows, digest, font=r"\small"):
    return "\n".join(
        [
            "% Auto-generated from audited summary.json; do not edit numeric cells.",
            f"% Source SHA256: {digest}",
            "% Regenerate with experiments/export_single_seed_paper_tables.py.",
            r"\begin{table*}[tb]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            font,
            r"\setlength{\tabcolsep}{4pt}",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            row(headers),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


def main():
    raw = SUMMARY.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    study = json.loads(raw)
    assert study["seed"] == 20261011
    assert not study["pending"]
    runs = {(r["workload"], r["ga"], r["arm"]): r for r in study["runs"]}
    assert len(runs) == 15
    winners = {}
    for group in study["retiming"]:
        assert group["status"] == "completed"
        for arm in group["arms"]:
            key = (group["workload"], group["ga"], arm["name"])
            assert len(arm["samples_us"]) == 30
            assert statistics.median(arm["samples_us"]) == arm["median_us"]
            assert len(arm["checks"]) == 4 and all(c["passed"] for c in arm["checks"])
            assert arm["result_sha256"] == runs[key]["result_sha256"]
            winners[key] = arm
    assert len(winners) == 15
    for run in runs.values():
        assert run["seed"] == study["seed"]
        assert run["budget"]["actual_submitted"] == 32
        assert run["budget"]["budget_completed"]
        assert run["outcomes"]["finite"] == 32

    primary = []
    for index, (workload, ga, name) in enumerate(GROUPS):
        if index:
            primary.append(r"\midrule")
        for arm in ARMS:
            key = (workload, ga, arm)
            run = runs[key]
            primary.append(
                row(
                    (
                        name if arm == "B" else "",
                        arm,
                        run["generated"],
                        run["verify_failed"],
                        f'{run["phases_s"]["search_s"]:.3f}',
                        f'{run["tune_s"]:.3f}',
                        f'{run["best_us"]:.4f}',
                        f'{winners[key]["median_us"]:.4f}',
                    )
                )
            )
    primary_tex = table(
        "同预算主结果：预先固定一个搜索 seed（20261011），每臂 32 submitted，"
        "均得到 32 次有限且正确的测量。B=baseline，C=CS+H，D=CS+H+Draft-GA。"
        "观测最优来自调优数据库；重测固定原 top-1，每臂 30 个技术重复的中位数，"
        "不回填调优曲线，也不增加独立搜索 seed。",
        "tab:single-primary",
        "llrrrrrr",
        (
            "算子 / GA",
            "臂",
            "生成",
            r"\shortstack{Verify\\失败}",
            r"\shortstack{Search\\(s)}",
            r"\shortstack{Tune\\(s)}",
            r"\shortstack{观测最优\\($\mu$s)}",
            r"\shortstack{原 top-1 重测\\中位数 ($\mu$s)}",
        ),
        primary,
        digest,
        font=r"\fontsize{9}{11}\selectfont",
    )

    ga_rows = []
    for ga in (1, 4):
        before = runs[("matmul128", ga, "C")]
        after = runs[("matmul128", ga, "D")]
        saved = before["xgb_prediction_s"] - after["xgb_prediction_s"]
        tune_reduction = 100 * (1 - after["tune_s"] / before["tune_s"])
        ga_rows.append(
            row(
                (
                    ga,
                    rf'${before["xgb_inputs"]}\to{after["xgb_inputs"]}$',
                    f'{before["xgb_prediction_s"]:.6f}',
                    f'{after["xgb_prediction_s"]:.6f}',
                    f"{saved:.6f}",
                    f'{after["draft_scoring_s"]:.6f}',
                    f"{tune_reduction:.2f}",
                )
            )
        )
    ga_tex = table(
        "MatMul128 的 GA 密度与评分开销：同一搜索 seed，每臂 32 submitted。"
        "预测节省与 Tune 减少均比较 C（CS+H）到 D（CS+H+Draft-GA）；"
        "粗筛自身评分和 XGBoost 预测已嵌套于 Search，不能重复加入总 Tune。",
        "tab:single-ga",
        "rrrrrrr",
        (
            "GA",
            r"\shortstack{XGB 输入\\C$\to$D}",
            r"\shortstack{C 预测\\(s)}",
            r"\shortstack{D 预测\\(s)}",
            r"\shortstack{预测节省\\(s)}",
            r"\shortstack{D 粗筛评分\\(s)}",
            r"\shortstack{Tune 减少\\(\%)}",
        ),
        ga_rows,
        digest,
    )

    target_rows = []
    target_count = 0
    for index, (workload, ga, name) in enumerate(GROUPS):
        if index:
            target_rows.append(r"\midrule")
        group = study["targets"][f"{workload}_ga{ga}"]
        for point_index, (threshold, arms) in enumerate(
            sorted(group.items(), key=lambda item: float(item[0]))
        ):
            cells = []
            for arm in ARMS:
                point = arms[arm]
                target_count += 1
                if point["attained"]:
                    cells.append(f'{point["submitted"]} / {point["tune_s"]:.3f}')
                else:
                    cells.append(f'NR({point["observed_until_s"]:.3f})')
            target_rows.append(
                row((name if point_index == 0 else "", f"{float(threshold):.4f}", *cells))
            )
    assert target_count == 39
    targets_tex = table(
        "全部目标阈值的达标提交数 / 完整 Tune 秒数：一个搜索 seed，每臂 32 submitted。"
        "NR(截止秒数) 表示预算耗尽仍未达标。MatMul128 的 40/50\,$\mu$s 为预定目标；"
        "扩展算子的三档目标为各自 baseline 调优终点的 1/1.01/1.05 倍，属于回顾性描述。"
        "整批 callback 完成后才判定达标，不插值，准备时间单列；"
        "目标依据调优观测，独立原赢家重测不回填本表。",
        "tab:single-targets",
        "lrrrr",
        (
            "算子 / GA",
            r"\shortstack{目标\\($\mu$s)}",
            r"\shortstack{B\\提交 / Tune (s)}",
            r"\shortstack{C\\提交 / Tune (s)}",
            r"\shortstack{D\\提交 / Tune (s)}",
        ),
        target_rows,
        digest,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "single_seed_primary_20261010.tex": primary_tex,
        "single_seed_ga_20261010.tex": ga_tex,
        "single_seed_targets_20261010.tex": targets_tex,
    }
    for filename, tex in outputs.items():
        destination = OUTPUT / filename
        destination.write_text(tex, encoding="utf-8")
        print(destination.relative_to(ROOT))
    print(f"Audited {len(runs)} runs, {len(winners)} fixed winners, {target_count} target points.")


if __name__ == "__main__":
    main()
