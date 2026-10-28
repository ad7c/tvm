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
"""Improve paper label legibility without changing the archived pilot data.

Run .venv-tune/python.exe experiments/plot_candidate_paper_layout_20261008.py.
This is offline plotting only: no TVM import, compilation or GPU measurement.
All original figures, scripts and manifests remain untouched.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import plot_paper_pilot_20261008 as pilot


OUTPUT = pilot.ROOT / "papers/figures/pilot_layout_20261008"
LABEL_MINIMUM = 140
PALETTE = ["#009E73", "#D55E00", "#E69F00", "#CC79A7", "#56B4E9", "#0072B2", "#999999"]


def main():
    runs, identity = pilot.load_runs()
    pilot.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                              "axes.spines.top": False, "axes.spines.right": False,
                              "svg.fonttype": "none", "svg.hashsalt": "pilot-paper-layout-20261008"})
    figure, axes = pilot.plt.subplots(1, 2, figsize=(9.6, 5.8), sharey=True)
    records = []
    for axis, seed in zip(axes, (20260931, 20260932)):
        arms = ("Baseline", "CS+H")
        results = [runs[seed, arm]["result"] for arm in arms]
        counts = [pilot.outcome_counts(result) for result in results]
        bottom = pilot.np.zeros(2)
        for category, color in zip(counts[0], PALETTE):
            heights = pilot.np.array([value[category] for value in counts])
            if not heights.any():
                continue
            axis.bar([0, 1], heights, bottom=bottom, width=0.58, label=category,
                     color=color, edgecolor="white", linewidth=0.5)
            for index, height in enumerate(heights):
                if height >= LABEL_MINIMUM:
                    axis.text(index, bottom[index] + height / 2, str(height), ha="center", va="center",
                              fontsize=10, color="white" if color in ("#009E73", "#D55E00") else "black")
            bottom += heights
        for index, (arm, result) in enumerate(zip(arms, results)):
            accounting = result["search"]["totals"]["accounting"]
            pilot.require(bottom[index] == accounting["generated"], "Plotted outcome conservation")
            ratio = accounting["passed"] / accounting["generated"] * 100
            axis.text(index, accounting["generated"] + 25,
                      f"{accounting['generated']} total\n{ratio:.1f}% passed", ha="center", fontsize=10)
            records.append({"seed": seed, "arm": arm,
                            "result_path": runs[seed, arm]["path"].relative_to(pilot.ROOT).as_posix(),
                            "outcome_counts": counts[index], "total": accounting["generated"],
                            "passed_percent": ratio,
                            "inside_bar_labels": {key: value for key, value in counts[index].items()
                                                  if value >= LABEL_MINIMUM}})
        axis.set_title(f"Seed {seed}", fontsize=11)
        axis.set_xticks([0, 1], arms)
        axis.set_ylim(0, 1530)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Candidate attempts (not unique traces)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=3,
                  frameon=False, fontsize=9)
    figure.suptitle("Pilot: candidate outcomes under the same 16-measurement budget", fontsize=12)
    figure.tight_layout(rect=(0, 0.21, 1, 0.94))
    figure.text(0.5, 0.015,
                "Passed = postprocessing passed, not a GPU measurement or a unique candidate.\n"
                "All segments retained; small-segment counts are in the manifest. Trace unknown is not a final outcome.",
                ha="center", va="bottom", fontsize=8, color="#444444")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("png", "svg"):
        path = OUTPUT / f"pilot_candidate_outcomes.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white",
                       metadata={"Creator": Path(__file__).name})
        raw = path.read_bytes()
        outputs[path.name] = {"sha256": pilot.digest_bytes(raw), "bytes": len(raw)}
    pilot.plt.close(figure)
    scripts = {}
    for path in (Path(__file__), Path(pilot.__file__)):
        raw = path.read_bytes()
        scripts[path.relative_to(pilot.ROOT).as_posix()] = {
            "sha256": pilot.digest_bytes(raw), "bytes": len(raw)}
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Label-layout-only revision of the existing two-seed candidate-outcome pilot figure",
        "scripts": scripts, "inputs": pilot.INPUTS, "outputs": outputs,
        "shared_identity_and_config": identity, "plotted_runs": records,
        "inside_bar_label_minimum": LABEL_MINIMUM,
        "palette_by_outcome": dict(zip(records[0]["outcome_counts"], PALETTE)),
        "validation": {"passed": True, "audited_input_runs": len(runs), "plotted_runs": len(records),
                       "checks": ["Original load_runs audits identities, budgets, callbacks and accounting.",
                                  "Original outcome_counts defines the unchanged mutually exclusive partition.",
                                  "Every plotted stacked total equals generated candidate attempts."]},
        "notes": ["No data, colors or plotted segment heights changed; only text size and small labels changed.",
                  "All small-category counts are retained in plotted_runs, not silently dropped.",
                  "Pilot observations only; no significance, unique-trace count or stable speedup claim.",
                  "Passed means postprocessing passed, not independently validated numerical correctness."]}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(runs)} archived runs; plotted {len(records)} runs to {OUTPUT}")


if __name__ == "__main__":
    main()
