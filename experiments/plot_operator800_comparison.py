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
"""Plot complete operator arms before proceeding to the next workload."""

import argparse
import csv
import json
from pathlib import Path

from operator800_cohort import collect_cohort
from report_single_seed_study import COLORS, LABELS, digest, target_point
from watch_operator800_study import write_json_atomic


def target_rows(rows, ratios):
    baseline = next(row for row in rows if row["arm"] == "B")
    return [{"arm": row["arm"], "ratio": ratio, "target_us": baseline["best_us"] * ratio,
             **target_point(row, baseline["best_us"] * ratio)} for ratio in ratios for row in rows]


def plot_comparison(manifest_path, workload, output):
    report = collect_cohort(manifest_path)
    rows = sorted((row for row in report["runs"] if row["workload"] == workload), key=lambda row: row["arm"])
    if len(rows) < 2 or not any(row["arm"] == "B" for row in rows):
        raise ValueError("Comparison requires the complete baseline and another complete arm")
    if any(row["budget"]["actual_submitted"] != 800 for row in rows):
        raise ValueError("Comparison requires full 800-submission arms")
    output.mkdir(parents=True, exist_ok=True)
    targets = target_rows(rows, (1.0, 1.01, 1.05))
    selected = {"workload": workload, "seed": report["seed"], "arms": [row["arm"] for row in rows],
                "runs": rows, "targets": targets, "independent_retiming_used": False,
                "protocol_sha256": report["protocol_sha256"], "input_sha256": report["input_sha256"]}
    write_json_atomic(output / "summary.json", selected)
    with (output / "convergence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "submitted", "finite", "best_us", "tune_s"])
        writer.writeheader()
        for row in rows:
            for point in row["curve"]:
                writer.writerow({"arm": row["arm"], **point})
    with (output / "targets.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    note = "800 submissions per arm; one seed; measured batch endpoints. Independent winner retiming not included."

    def save(fig, name):
        fig.text(0.5, 0.012, note, ha="center", fontsize=8, color="0.35")
        fig.tight_layout(rect=(0, 0.055, 1, 0.94))
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / (name + "." + extension), dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for row in rows:
        points = [point for point in row["curve"] if point["best_us"] is not None]
        for axis, key, divisor in ((axes[0], "submitted", 1), (axes[1], "tune_s", 60)):
            end = 800 if key == "submitted" else row["tune_s"] / divisor
            axis.step([point[key] / divisor for point in points] + [end],
                      [point["best_us"] for point in points] + [row["best_us"]],
                      where="post", color=COLORS[row["arm"]], label=LABELS[row["arm"]], linewidth=1.8)
    axes[0].set_xlabel("Submitted candidates (failures included)")
    axes[1].set_xlabel("Full tuning wall time (min)")
    for axis in axes:
        axis.set_ylabel("Best measured latency (us)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle(f"{workload} | search convergence | seed {report['seed']}")
    save(fig, "convergence")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, ratio in zip(axes, (1.0, 1.01, 1.05)):
        group = [target for target in targets if target["ratio"] == ratio]
        largest = max(row["tune_s"] for row in rows) / 60
        for index, target in enumerate(group):
            if target["attained"]:
                value = target["tune_s"] / 60
                axis.bar(index, value, width=0.55, color=COLORS[target["arm"]])
                axis.text(index, value + largest * 0.025, f"{value:.1f}", ha="center", fontsize=9)
            else:
                axis.text(index, largest * 0.45, "Not reached\nin 800", ha="center", fontsize=9, color="0.4")
        axis.set_xticks(range(len(rows)), [LABELS[row["arm"]] for row in rows], rotation=15, ha="right")
        axis.set_ylim(0, largest * 1.15)
        axis.set_ylabel("Time to target (min)")
        axis.set_title(f"Baseline final x {ratio:.2f}\n{group[0]['target_us']:.3f} us")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{workload} | time to predeclared targets")
    save(fig, "time_to_target")

    fig, axis = plt.subplots(figsize=(8.5, 3.8))
    keys = ["search_s", "builder_s", "runner_and_notify_s", "callbacks_s", "residual_s"]
    names = ["Search", "Builder", "Runner + notify", "Callbacks", "Residual"]
    colors = ["#56B4E9", "#0072B2", "#009E73", "#E69F00", "#BBBBBB"]
    bottom = [0.0] * len(rows)
    for key, name, color in zip(keys, names, colors):
        values = [row["phases_s"][key] / 60 for row in rows]
        axis.bar([LABELS[row["arm"]] for row in rows], values, bottom=bottom, label=name, color=color)
        bottom = [left + right for left, right in zip(bottom, values)]
    axis.set_ylabel("Full tuning wall time (min)")
    axis.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{workload} | non-overlapping tuning phases")
    save(fig, "phases")

    body = [f"# {workload} 完整组对比", "", "比较组：" + "、".join(LABELS[row["arm"]] for row in rows),
            "每组 800 次真实提交，单 seed，GA 1 代。搜索实测值，未混入独立赢家重测。", "",
            "| 方法 | 最佳实测 µs | 总调优 min | 首次观察最终最佳的提交数 | 首次观察时间 min | 模型累计样本 |",
            "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        hit = row["own_best"]
        body.append(f"| {LABELS[row['arm']]} | {row['best_us']:.6f} | {row['tune_s']/60:.2f} | "
                    f"{hit['submitted']} | {hit['tune_s']/60:.2f} | {row['xgb_data_size']} |")
    body += ["", "| 目标倍率（相对于 baseline 搜索最终延迟） | 方法 | 是否达到 | 提交数 | 时间 min |",
             "|---:|---|---|---:|---:|"]
    for target in targets:
        body.append(f"| {target['ratio']:.2f} | {LABELS[target['arm']]} | "
                    + (f"是 | {target['submitted']} | {target['tune_s']/60:.2f} |" if target["attained"] else "否 | — | — |"))
    body += ["", "严格目标与 +1%、+5% 范围全部呈现，未达到目标不填虚构时间。",
             "训练样本随各自反馈累计；各方法模型更新与搜索评分开销保存在 summary.json。",
             "收敛曲线按实际批次反馈时间绘制，失败候选计入提交预算。"]
    if len(rows) < 3:
        body += ["", "当前只含已完成的两个组；第三组完成后另生成 B/C/D 三组对比目录。"]
    (output / "RESULTS.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    write_json_atomic(output / "manifest.json", {"plot_script_sha256": digest(Path(__file__)),
        "workload": workload, "arms": selected["arms"], "input_sha256": selected["input_sha256"],
        "protocol_sha256": report["protocol_sha256"], "independent_retiming_used": False,
        "output_sha256": {path.name: digest(path) for path in output.iterdir()
                          if path.is_file() and path.name != "manifest.json"}})
    return {"workload": workload, "arms": selected["arms"], "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot_comparison(args.manifest.resolve(), args.workload, args.output.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
