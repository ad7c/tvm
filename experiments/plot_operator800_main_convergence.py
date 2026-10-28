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
"""Create trimmed convergence views and first-100-submission zooms from frozen data.

Only offline analysis is performed. This module does not import TVM or Torch,
launch tuning, interpolate results, pad short runs, or use retimed observations.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "papers/figures/operator800_20261011_cohort/summary.json"
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "paper_main_convergence_zoomed"
DISPLAY_SUBMISSIONS = {
    "matmul128": 500,
    "matmul256": 600,
    "conv2d_nchw3x3": 700,
    "reduction4096": 100,
}
WORKLOADS = (
    ("matmul128", "(a) MatMul128"),
    ("matmul256", "(b) MatMul256"),
    ("conv2d_nchw3x3", "(c) Conv2D"),
    ("reduction4096", "(d) Reduction4096"),
)
STYLES = {
    "B": {"color": "#0072B2", "linestyle": "-", "marker": "o", "label": "B: MetaSchedule"},
    "C": {"color": "#009E73", "linestyle": "--", "marker": "s", "label": "C: CS+H"},
    "D": {"color": "#D55E00", "linestyle": "-.", "marker": "^", "label": "D: Scout"},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(source):
    report = json.loads(source.read_text(encoding="utf-8"))
    selected = {}
    output_rows = []
    audit = []
    for workload, _ in WORKLOADS:
        for arm in "BCD":
            runs = [r for r in report["runs"] if r["workload"] == workload and r["arm"] == arm]
            if len(runs) != 1:
                raise ValueError(f"Expected one frozen run for {workload}/{arm}, found {len(runs)}")
            run = runs[0]
            points = []
            skipped = []
            previous = None
            for index, point in enumerate(run["curve"]):
                value = point["best_us"]
                if value is None or not math.isfinite(value):
                    skipped.append(index)
                    continue
                if value <= 0 or not math.isfinite(point["tune_s"]):
                    raise ValueError(f"Invalid recorded observation {workload}/{arm}/{index}")
                if previous is not None and (
                    point["submitted"] < previous["submitted"]
                    or point["tune_s"] < previous["tune_s"]
                    or value > previous["best_us"] + 1e-10
                ):
                    raise ValueError(f"Nonmonotone source curve {workload}/{arm}/{index}")
                row = {
                    "workload": workload,
                    "arm": arm,
                    "source_curve_index": index,
                    "submitted": point["submitted"],
                    "finite": point["finite"],
                    "tune_s": point["tune_s"],
                    "tune_min": point["tune_s"] / 60.0,
                    "best_us": value,
                }
                output_rows.append(row)
                points.append(point)
                previous = point
            if not points:
                raise ValueError(f"No observed point for {workload}/{arm}")
            actual = run["budget"]["actual_submitted"]
            if points[-1]["submitted"] != actual:
                raise ValueError(f"Final feedback count differs for {workload}/{arm}")
            if abs(points[-1]["best_us"] - run["best_us"]) > 1e-10:
                raise ValueError(f"Endpoint differs from frozen best for {workload}/{arm}")
            if workload == "reduction4096" and actual != {"B": 85, "C": 88, "D": 88}[arm]:
                raise ValueError("Unexpected short-space submission count")
            selected[(workload, arm)] = (run, points)
            audit.append({
                "workload": workload,
                "arm": arm,
                "id": run["id"],
                "raw_result_path": run["result_path"],
                "raw_result_sha256": run["result_sha256"],
                "source_points": len(run["curve"]),
                "plotted_points": len(points),
                "omitted_null_or_nonfinite_indices": skipped,
                "first_point": points[0],
                "last_point": points[-1],
                "actual_submitted": actual,
                "terminal_status": run["budget"]["terminal_status"],
                "resumption": run.get("resumption"),
            })
    return report, selected, output_rows, audit


def plot(source, output):
    report, runs, coordinate_rows, audit = collect(source)
    output.mkdir(parents=True, exist_ok=True)
    coordinates = output / "convergence_coordinates.csv"
    with coordinates.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coordinate_rows[0]))
        writer.writeheader()
        writer.writerows(coordinate_rows)

    # Conda on Windows resolves the BLAS delay-load dependencies from Library/bin.
    # Keep the DLL-directory handle alive through figure rendering.
    dll_directory = Path(sys.prefix) / "Library/bin"
    dll_handle = (
        os.add_dll_directory(str(dll_directory))
        if os.name == "nt" and dll_directory.is_dir()
        else None
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(4, 3, figsize=(6.3, 7.0), sharey="row")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.095, top=0.905,
                        hspace=0.66, wspace=0.16)
    handles = [Line2D([], [], color=STYLES[a]["color"], linestyle=STYLES[a]["linestyle"],
                      linewidth=1.35, label=STYLES[a]["label"]) for a in "BCD"]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.992),
               ncol=3, frameon=False, fontsize=8, columnspacing=1.5, handlelength=2.9)
    for column, heading in enumerate(("By submissions", "First 100 submissions", "By tuning time")):
        position = axes[0, column].get_position()
        fig.text((position.x0 + position.x1) / 2, 0.943, heading,
                 ha="center", va="center", fontsize=7.6, weight="bold")
    rendering_audit = []
    display_windows = []
    for row, (workload, title) in enumerate(WORKLOADS):
        submission_limit = DISPLAY_SUBMISSIONS[workload]
        # Pick one time limit for all arms, after each has reached the submission
        # display boundary. This preserves all improvements without arm-wise clocks.
        boundary_times = []
        last_improvements = {}
        for arm in "BCD":
            run, points = runs[(workload, arm)]
            boundary = next((p for p in points if p["submitted"] >= submission_limit), points[-1])
            boundary_times.append(boundary["tune_s"] / 60)
            best = next(p for p in points if p["best_us"] == run["best_us"])
            if best["submitted"] > submission_limit:
                raise AssertionError(f"Display crop would hide a later improvement: {workload}/{arm}")
            last_improvements[arm] = {
                "submitted": best["submitted"], "tune_min": best["tune_s"] / 60,
            }
        time_limit = math.ceil(max(boundary_times) / 5) * 5
        assert all(p["tune_min"] <= time_limit for p in last_improvements.values())
        display_windows.append({"workload": workload, "submitted_limit": submission_limit,
                                "zoom_limit": 100, "time_limit_min": time_limit,
                                "last_improvements": last_improvements})
        ymin = min(p["best_us"] for arm in "BCD" for p in runs[(workload, arm)][1])
        ymax = max(p["best_us"] for arm in "BCD" for p in runs[(workload, arm)][1])
        margin = max(0.35, (ymax - ymin) * 0.075)
        for column, key, scale, limit in (
            (0, "submitted", 1, submission_limit),
            (1, "submitted", 1, 100),
            (2, "tune_s", 60.0, time_limit),
        ):
            axis = axes[row, column]
            for arm in "BCD":
                run, points = runs[(workload, arm)]
                style = STYLES[arm]
                x = [p[key] / scale for p in points]
                y = [p["best_us"] for p in points]
                line, = axis.step(x, y, where="post", color=style["color"],
                                  linestyle=style["linestyle"], linewidth=1.25,
                                  solid_capstyle="butt", zorder=3)
                # Actual endpoints only: a display crop is never marked as a run end.
                if x[-1] <= limit:
                    axis.plot(x[-1], y[-1], marker=style["marker"], markersize=3.3,
                              markerfacecolor="white", markeredgewidth=0.8,
                              markeredgecolor=style["color"], linestyle="none", zorder=4,
                              clip_on=False)
                # Verify the Matplotlib source coordinates against every frozen observation.
                if list(line.get_xdata()) != x or list(line.get_ydata()) != y:
                    raise AssertionError("Plot data differs from frozen coordinates")
                rendering_audit.append({"workload": workload, "arm": arm, "column": column,
                                        "drawstyle": line.get_drawstyle(), "points": len(points),
                                        "display_limit": limit,
                                        "visible_observations": sum(v <= limit for v in x)})
            axis.set_ylim(ymin - margin, ymax + margin)
            if column == 0:
                axis.set_title(title, loc="left", pad=4, weight="normal")
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            axis.tick_params(axis="both", direction="out", length=2.5, width=0.6, pad=2)
            axis.grid(axis="y", color="#D9DDE2", linewidth=0.55, alpha=0.85, zorder=0)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_xlim(0, limit)
            if column == 0:
                axis.set_ylabel("Best measured latency ($\\mu$s)", labelpad=4)
            else:
                axis.tick_params(axis="y", left=False, labelleft=False)
            if column < 2:
                axis.set_xlabel("Submitted candidates", labelpad=2, fontsize=7)
                ticks = [0, 25, 50, 75, 100] if limit == 100 else (
                    [0, 100, 200, 300, 400, 500] if limit == 500 else
                    [0, 200, 400, 600] if limit == 600 else [0, 200, 400, 600, 700])
                axis.set_xticks(ticks)
            else:
                locator = MaxNLocator(nbins=4, integer=True)
                ticks = [float(v) for v in locator.tick_values(0, limit) if 0 <= v <= limit]
                if ticks[-1] != limit:
                    if limit - ticks[-1] < limit * 0.15:
                        ticks.pop()
                    ticks.append(limit)
                axis.set_xticks(ticks)
                axis.set_xlabel("Active Tune time (min)", labelpad=2, fontsize=7)
    fig.text(0.545, 0.043, "Display crops only; all observed improvements retained in left/right panels.",
             ha="center", va="center", fontsize=6.6, color="#444444")
    fig.text(0.545, 0.021, "Reduction4096: 85/88/88 submissions (B/C/D); middle panel shows the full run.",
             ha="center", va="center", fontsize=6.6, color="#444444")
    paths = []
    for extension in ("pdf", "svg", "png"):
        path = output / f"operator800_main_convergence.{extension}"
        fig.savefig(path, dpi=240, metadata={"Creator": Path(__file__).name} if extension == "pdf" else None)
        paths.append(path)
    plt.close(fig)
    if dll_handle is not None:
        dll_handle.close()
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": digest(source),
        "source_protocol_sha256": report.get("protocol_sha256"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": digest(__file__),
        "python": sys.version,
        "matplotlib": matplotlib.__version__,
        "seed": report["seed"],
        "figure_size_in": [6.3, 7.0],
        "panels": "four operator rows by trimmed submissions, first 100 submissions, trimmed active time; shared linear y within each row",
        "display_windows": display_windows,
        "styles": STYLES,
        "coordinate_csv": {"path": str(coordinates), "sha256": digest(coordinates), "rows": len(coordinate_rows)},
        "figures": [{"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size} for path in paths],
        "runs": audit,
        "rendering_audit": rendering_audit,
        "policies": [
            "Only source feedback-batch points with finite non-null positive best_us are plotted.",
            "All finite observations remain in CSV and Matplotlib source coordinates; axes alone crop the view.",
            "No synthetic initial point, interpolation, smoothing, or padding; step-post displays the incumbent between feedback points.",
            "The same submission/time window is used for B/C/D within each operator; outer panels preserve every improvement.",
            "Post-hoc display cropping does not imply an online stopping rule; full-budget figure is retained separately.",
            "Hollow markers denote actual endpoints only when visible; cropped boundaries are never marked as termination.",
            "Independent fixed-winner remeasurement never replaces search-curve observations.",
            "Reduction stops at the actual 85/88/88 submissions; the 800 cap is not treated as attained.",
            "Conv2D C uses the report's verified history-continuation active Tune clock; pause and reconstruction are not added.",
            "Time coordinates are converted from seconds to minutes without any other adjustment.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "source_curves": len(audit),
                      "coordinate_rows": len(coordinate_rows), "files": [str(p) for p in paths]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.input.resolve(), args.output.resolve())
