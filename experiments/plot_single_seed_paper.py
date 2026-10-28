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
"""Make paper-sized figures from the audited report; never run a search."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
GROUPS = [("matmul128", 1), ("matmul128", 4), ("matmul256", 1),
          ("conv2d_nchw3x3", 1), ("reduction4096", 1)]
NAMES = {"matmul128": "MatMul128", "matmul256": "MatMul256",
         "conv2d_nchw3x3": "Conv2D", "reduction4096": "Reduction"}
SHORT = {"matmul128": "M128", "matmul256": "M256",
         "conv2d_nchw3x3": "Conv", "reduction4096": "Red"}
COLORS = {"B": "#0072B2", "C": "#009E73", "D": "#E69F00"}
STYLES = {"B": "-", "C": "--", "D": "-."}
LABELS = {"B": "B: Baseline", "C": "C: CS+H", "D": "D: CS+H+Draft-GA"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_rows(report, group):
    rows = {r["arm"]: r for r in report["runs"]
            if (r["workload"], r["ga"]) == group}
    if set(rows) != set("BCD"):
        raise ValueError(f"Incomplete group: {group}")
    return rows


def title(group):
    return f"{NAMES[group[0]]} (GA={group[1]})"


def panel_grid(plt, height=4.05):
    fig, grid = plt.subplots(2, 3, figsize=(6.25, height))
    axes = list(grid.flat)
    axes[-1].axis("off")
    return fig, axes


def legend_panel(axis, plt, note):
    handles = [plt.Line2D([0], [0], color=COLORS[a], linestyle=STYLES[a],
                         marker="o", markersize=3, label=LABELS[a]) for a in "BCD"]
    axis.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
                handlelength=2.0, labelspacing=0.65, borderaxespad=0)
    axis.text(0.0, 0.40, note, transform=axis.transAxes, va="top", fontsize=8,
              color="#333333", linespacing=1.5)


def save(fig, output, name, plt, figures):
    for extension in ("pdf", "png", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
    figures.append({"name": name, "width_inches": float(fig.get_figwidth()),
                    "height_inches": float(fig.get_figheight())})
    plt.close(fig)


def plot(report, output):
    # This process is launched under run_single_seed_study.environment().
    # Also establish the same native-library profile before importing plotting.
    if os.name == "nt":
        os.environ["PATH"] = str(Path(sys.executable).parent / "Library/bin") + os.pathsep + os.environ["PATH"]
        os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8.7, "axes.labelsize": 8.2,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6})
    figures = []
    fig, axes = panel_grid(plt)
    for index, (axis, group) in enumerate(zip(axes, GROUPS)):
        for arm, row in group_rows(report, group).items():
            points = [p for p in row["curve"] if p["best_us"] is not None]
            axis.step([p["submitted"] for p in points], [p["best_us"] for p in points],
                      where="post", color=COLORS[arm], linestyle=STYLES[arm],
                      marker="o", markersize=2.3, linewidth=1.2)
        axis.set_title(title(group), pad=6)
        axis.set_xlim(2.5, 33)
        axis.set_xticks([4, 16, 32])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.grid(alpha=0.18, linewidth=0.5)
        if index in (0, 3):
            axis.set_ylabel("Best observed latency (µs)")
        if index >= 3:
            axis.set_xlabel("Submitted candidates")
    legend_panel(axes[-1], plt, "One search seed.\nBatch endpoints only.\nTuning observations;\nindependent retiming\nshown separately.")
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.13, top=0.93,
                        wspace=0.45, hspace=0.61)
    save(fig, output, "f1_search_quality", plt, figures)

    fig, axes = panel_grid(plt)
    for index, (axis, group) in enumerate(zip(axes, GROUPS)):
        rows = group_rows(report, group)
        target = 40.0 if group[0] == "matmul128" else rows["B"]["best_us"] * 1.05
        axis.set_title(title(group) + f"\nTarget ≤ {target:.2f} µs", pad=6)
        upper = max(r["tune_s"] for r in rows.values()) * 1.24
        for y, arm in enumerate("BCD"):
            row = rows[arm]
            hit = next((p for p in row["curve"]
                        if p["best_us"] is not None and p["best_us"] <= target), None)
            if hit:
                seconds = hit["tune_s"]
                axis.barh(y, seconds, height=0.55, color=COLORS[arm], alpha=0.85)
                label = f"{seconds:.0f}s ({hit['submitted']})"
                axis.text(seconds + upper * 0.025, y, label, va="center", fontsize=8)
            else:
                seconds = row["tune_s"]
                axis.scatter(seconds, y, marker="x", s=38, color=COLORS[arm], linewidths=1.5)
                axis.text(seconds - upper * 0.03, y + 0.34,
                          f"NR ≥ {seconds:.0f}s", ha="right", va="center", fontsize=8)
        axis.set_yticks([0, 1, 2], ["B", "C", "D"])
        axis.set_ylim(2.8, -0.7)
        axis.set_xlim(0, upper)
        axis.set_xticks([0, 400, 800])
        axis.grid(axis="x", alpha=0.18, linewidth=0.5)
        if index >= 3:
            axis.set_xlabel("Full tune wall time (s)")
    legend_panel(axes[-1], plt, "Parentheses: submitted.\nNR: right-censored.\nM128: preset 40 µs.\nOther targets: baseline\nendpoint × 1.05\n(retrospective).")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.9,
                        wspace=0.45, hspace=0.69)
    save(fig, output, "f2_time_to_target", plt, figures)

    rows = [group_rows(report, group)[arm] for group in GROUPS for arm in "BCD"]
    fig, axis = plt.subplots(figsize=(6.25, 4.45))
    bottoms = [0.0] * len(rows)
    phases = [("search_s", "Search", "#0072B2"),
              ("builder_s", "Builder", "#E69F00"),
              ("runner_and_notify_s", "Runner + notify", "#009E73"),
              ("callbacks_s", "Callbacks", "#CC79A7"),
              ("residual_s", "Tune residual", "#B3B3B3")]
    for key, label, color in phases:
        values = [r["phases_s"][key] for r in rows]
        axis.barh(range(len(rows)), values, left=bottoms, height=0.67,
                  color=color, edgecolor="white", linewidth=0.3, label=label)
        bottoms = [before + value for before, value in zip(bottoms, values)]
    for i, row in enumerate(rows):
        axis.text(row["tune_s"] + 8, i, f"{row['tune_s']:.1f}", va="center", fontsize=8)
    for separator in (2.5, 5.5, 8.5, 11.5):
        axis.axhline(separator, color="#777777", linewidth=0.45, linestyle=":")
    axis.set_yticks(range(len(rows)),
                    [f"{SHORT[r['workload']]}/G{r['ga']} {r['arm']}" for r in rows])
    axis.set_ylim(len(rows) - 0.35, -0.65)
    axis.set_xlim(0, max(r["tune_s"] for r in rows) * 1.1)
    axis.set_xlabel("Full tune wall time (s)")
    axis.set_xticks([0, 200, 400, 600, 800])
    axis.grid(axis="x", alpha=0.18, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.51, 1.16), ncol=3,
                fontsize=8, frameon=False, columnspacing=1.4, handlelength=1.5)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.12, top=0.86)
    save(fig, output, "f3_stage_time", plt, figures)

    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.6))
    x = [0, 1]
    for arm, offset in (("C", -0.15), ("D", 0.15)):
        selected = [group_rows(report, ("matmul128", ga))[arm] for ga in (1, 4)]
        for axis, key in zip(axes, ("xgb_inputs", "xgb_prediction_s")):
            positions = [position + offset for position in x]
            values = [r[key] for r in selected]
            axis.bar(positions, values, width=0.27, color=COLORS[arm], label=arm, alpha=0.9)
            for position, value in zip(positions, values):
                label = f"{value:d}" if key == "xgb_inputs" else f"{value:.3f}"
                axis.text(position, value, label, ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("XGB prediction inputs")
    axes[0].set_ylim(0, 1480)
    axes[1].set_ylabel("XGB prediction time (s)")
    axes[1].set_ylim(0, 7.6)
    for axis in axes:
        axis.set_xticks(x, ["GA=1", "GA=4"])
        axis.set_xlim(-0.5, 1.5)
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)
        axis.set_axisbelow(True)
    axes[1].legend(loc="upper left", ncol=2, frameon=False, handlelength=1.2)
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.16, top=0.94, wspace=0.36)
    save(fig, output, "f4_ga_model_cost", plt, figures)

    fig, axes = panel_grid(plt)
    for index, (axis, group) in enumerate(zip(axes, GROUPS)):
        timing = next(t for t in report["retiming"]
                      if (t["workload"], t["ga"]) == group)
        for x, arm in enumerate("BCD"):
            data = next(a for a in timing["arms"] if a["name"] == arm)
            median, low, high = (data[k] for k in ("median_us", "p25_us", "p75_us"))
            if not low <= median <= high:
                raise ValueError("Invalid retiming quartiles")
            axis.errorbar(x, median, yerr=[[median - low], [high - median]],
                          fmt="o", color=COLORS[arm], markersize=4,
                          capsize=4, elinewidth=1.3, capthick=1.1)
        axis.set_title(title(group), pad=6)
        axis.set_xticks([0, 1, 2], ["B", "C", "D"])
        axis.set_xlim(-0.45, 2.45)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)
        axis.margins(y=0.2)
        if index in (0, 3):
            axis.set_ylabel("Retimed latency (µs)")
    legend_panel(axes[-1], plt, "Dots: median.\nWhiskers: P25–P75.\n30 samples per winner.\nFixed archived winners.\nTechnical repeats;\nnot search seeds.")
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.12, top=0.93,
                        wspace=0.45, hspace=0.61)
    save(fig, output, "f5_independent_top1", plt, figures)
    return figures


def render_qa(output, figures):
    """Render the actual PDFs and make a contact sheet for separate visual review."""
    sys.path.insert(0, str(ROOT / "tmp/pdf-tools"))
    import pypdfium2 as pdfium
    from PIL import Image, ImageDraw
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    sheet = Image.new("RGB", (1000, 900), "white")
    for i, figure in enumerate(figures):
        with pdfium.PdfDocument(str(output / f"{figure['name']}.pdf")) as document:
            if len(document) != 1:
                raise ValueError("Scientific figure must have one PDF page")
            page = document[0]
            bitmap = page.render(scale=2.2)
            rendered = bitmap.to_pil().convert("RGB")
            rendered.save(qa / f"{figure['name']}.png")
            rendered.thumbnail((490, 260))
            tile = Image.new("RGB", (500, 300), "#eeeeee")
            tile.paste(rendered, ((500 - rendered.width) // 2, 28))
            ImageDraw.Draw(tile).text((10, 8), figure["name"], fill="black")
            sheet.paste(tile, ((i % 2) * 500, (i // 2) * 300))
            bitmap.close()
            page.close()
    sheet.save(qa / "contact-sheet.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "papers/figures/single_seed_20261010/summary.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "papers/figures/single_seed_paper_20261010")
    args = parser.parse_args()
    source_hash = digest(args.input)
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if report["pending"] or len(report["runs"]) != 15 or len(report["retiming"]) != 5:
        raise ValueError("Only a complete audited study can produce the paper figures")
    args.output.mkdir(parents=True, exist_ok=True)
    figures = plot(report, args.output)
    render_qa(args.output, figures)
    if digest(args.input) != source_hash:
        raise ValueError("Report changed during plotting")
    manifest = {"input": str(args.input.resolve()), "input_sha256": source_hash,
                "generator": str(Path(__file__).resolve()), "generator_sha256": digest(__file__),
                "seed": report["seed"], "figures": figures,
                "minimum_font_points": 8, "visual_review": "pending",
                "files": {str(p.relative_to(args.output)): digest(p)
                          for p in sorted(args.output.rglob("*"))
                          if p.is_file() and p.name != "manifest.json"}}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "figures": figures,
                      "input_sha256": source_hash, "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
