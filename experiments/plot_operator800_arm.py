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
"""Draw audited per-arm measured convergence and cumulative model diagnostics."""

import argparse
import csv
import json
from pathlib import Path

from operator800_cohort import accept_tuning, load_tuning
from report_single_seed_study import digest, LABELS
from watch_operator800_study import read_json, write_json_atomic


def plot_arm(root, spec, protocol, output):
    reference = accept_tuning(root, spec, protocol)
    row = load_tuning(reference, spec, protocol)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "summary.json", row)
    curve = row["curve"]
    with (output / "convergence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    model = row["model_progress"]
    keys = ["submitted", "tune_s", "data_size", "last_train_size", "update_s", "boosted_rounds",
            "train_rmse", "train_peak32", "next_batch_rmse", "next_batch_peak32"]
    with (output / "model_training.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(model)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    points = [point for point in curve if point["best_us"] is not None]
    hit = row["own_best"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for axis, key, divisor, label, limit in [
        (axes[0], "submitted", 1, "Submitted candidates (failures included)", spec["trials"]),
        (axes[1], "tune_s", 60, "Elapsed tuning time (min)", row["tune_s"] / 60),
    ]:
        xs = [point[key] / divisor for point in points]
        ys = [point["best_us"] for point in points]
        # Extend the final observed step to the end of the actual tuning call.
        axis.step(xs + [limit], ys + [ys[-1]], where="post", color="#0072B2", linewidth=1.8,
                  label=LABELS[row["arm"]])
        improvements = [point for i, point in enumerate(points)
                        if i == 0 or point["best_us"] < points[i - 1]["best_us"]]
        axis.scatter([point[key] / divisor for point in improvements],
                     [point["best_us"] for point in improvements], s=18, color="#0072B2", zorder=3)
        xbest = hit[key] / divisor
        axis.scatter([xbest], [row["best_us"]], marker="*", s=150, color="#D55E00", zorder=4)
        axis.annotate(f"{row['best_us']:.2f} us\nfirst observed at\n"
                      + (f"{hit['submitted']} submissions" if key == "submitted" else f"{xbest:.1f} min"),
                      xy=(xbest, row["best_us"]), xytext=(12, 22), textcoords="offset points",
                      fontsize=9, color="#A63B00", arrowprops={"arrowstyle": "-", "color": "#D55E00"})
        axis.set(xlabel=label, ylabel="Best measured latency (us)", xlim=(0, limit))
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", frameon=False)
    fig.suptitle(f"{spec['workload']} | {LABELS[row['arm']]} | seed {row['seed']} | {spec['trials']} submissions")
    fig.text(0.5, 0.015, "Observed search measurements at feedback-batch completion; independent winner retiming pending.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / ("convergence." + extension), dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    xs = [point["submitted"] for point in model]
    axes[0].step(xs, [point["data_size"] for point in model], where="post", color="#0072B2")
    axes[0].plot(xs, xs, linestyle=":", color="0.5", linewidth=1, label="Submitted")
    axes[0].set_ylabel("Cumulative training samples")
    axes[0].legend(frameon=False)
    axes[1].plot(xs, [point["train_rmse"] for point in model], color="#009E73", label="Training fit")
    axes[1].plot(xs, [point["next_batch_rmse"] for point in model], color="#D55E00", alpha=0.8,
                 label="New batch, before update")
    axes[1].set_ylabel("XGBoost p-RMSE")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(xs, [point["update_s"] for point in model], color="#0072B2")
    axes[2].set_ylabel("Cost-model callback time (s)")
    for axis in axes:
        axis.set_xlabel("Submitted candidates")
        axis.set_xlim(0, spec["trials"])
        axis.grid(alpha=0.2)
    fig.suptitle(f"{spec['workload']} | {LABELS[row['arm']]} | accumulated feedback")
    fig.text(0.5, 0.01, "Training fit and selected new-batch predictions are separate; update time includes features and validation.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / ("model_training." + extension), dpi=180)
    plt.close(fig)

    milestones = []
    for cutoff in (100, 200, 400, 600, 800):
        eligible = [point for point in curve if point["submitted"] <= cutoff]
        if eligible:
            point = eligible[-1]
            milestones.append(f"| {cutoff} | {point['submitted']} | {point['best_us']:.3f} | {point['tune_s'] / 60:.2f} |")
    phases = "\n".join(f"| {key} | {value:.3f} |" for key, value in row["phases_s"].items())
    body = (f"# {spec['workload']} {LABELS[row['arm']]} 实测结果\n\n"
            f"- 实际提交：{row['budget']['actual_submitted']}/{spec['trials']}；有限测量：{row['outcomes'].get('finite', 0)}。\n"
            f"- 搜索最佳：{row['best_us']:.6f} μs；首次可见于 {hit['submitted']} 次提交、{hit['tune_s'] / 60:.2f} 分钟。\n"
            f"- 总调优：{row['tune_s']:.3f} 秒（{row['tune_s'] / 60:.2f} 分钟）；进程总用时含启动开销另记。\n"
            f"- 代价模型：{len(model)} 次反馈更新，累计 {row['xgb_data_size']} 个样本；历史未重置。\n"
            f"- XGBoost 搜索评分：{row['xgb_prediction_s']:.3f} 秒；模型更新 callback 合计 {sum(p['update_s'] for p in model):.3f} 秒。\n\n"
            "曲线只使用实际测量和批次结束时间；星号表示首次观察到原搜索最终最佳候选。独立 30 样本重测尚未完成。\n\n"
            "| 提交上限 | 最近已完成批次的提交数 | 最佳实测延迟 (μs) | 调优时间 (分钟) |\n"
            "|---:|---:|---:|---:|\n" + "\n".join(milestones)
            + "\n\n| 阶段 | 时间 (s) |\n|---|---:|\n" + phases + "\n")
    if reference.get("postcheck_audit_path"):
        body += ("\n原始搜索已完成，但冻结脚本对小批次的末尾审核误报失败（原退出码 1 保留）。"
                 "独立审核重新核对提交数、数据库、批次时间、正确性、GA 评分架构、累计模型训练和冻结协议后通过。"
                 "审核派生视图与原文件分开存储，没有改写原始结果或退出记录。\n")
    (output / "RESULTS.md").write_text(body, encoding="utf-8")
    write_json_atomic(output / "manifest.json", {"plot_script_sha256": digest(Path(__file__)),
        "accepted_tuning": reference, "independent_retiming": "pending",
        "output_sha256": {path.name: digest(path) for path in output.iterdir()
                          if path.is_file() and path.name != "manifest.json"}})
    return {"id": spec["id"], "best_us": row["best_us"], "own_best": hit,
            "tune_s": row["tune_s"], "xgb_data_size": row["xgb_data_size"], "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = read_json(root / "protocol.json")
    spec = next(spec for spec in protocol["order"] if spec["id"] == args.id)
    print(json.dumps(plot_arm(root, spec, protocol, args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
