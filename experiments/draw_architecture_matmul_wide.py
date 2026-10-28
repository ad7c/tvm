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
"""Draw the English, two-column MatMul architecture as a vector figure."""

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "papers/figures/architecture_matmul_wide"


def main():
    if os.name == "nt":
        os.environ["PATH"] = str(Path(sys.executable).parent / "Library/bin") + os.pathsep + os.environ["PATH"]
        os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyArrowPatch, Rectangle

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": False})
    sans = FontProperties(fname="C:/Windows/Fonts/arial.ttf")
    bold = FontProperties(fname="C:/Windows/Fonts/arialbd.ttf")
    mono = FontProperties(fname="C:/Windows/Fonts/consola.ttf")
    # Coordinates are deliberately 2.1x the intended printed point sizes.
    width, height = 984, 480
    printed_width = 468
    scale = printed_width / width
    fig = plt.figure(figsize=(printed_width / 72, height * scale / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")

    ink, edge = "#202D3B", "#647181"
    blue, blue_fill = "#245F91", "#E5F0F8"
    orange, orange_fill = "#A56519", "#FFF0D9"
    feedback = "#AB4545"
    texts = []

    def text(x, y, value, size=16, color=ink, align="left", font=None,
             weight=False, limit=None):
        artist = ax.text(x, y, value, fontsize=size * scale, color=color,
                         ha=align, va="top", fontproperties=font or (bold if weight else sans),
                         linespacing=1.1, zorder=6)
        texts.append((artist, limit))
        return artist

    def box(x, y, w, h, fill="white", stroke=edge, dashed=False, lw=1.3):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=stroke,
                               linewidth=lw * scale,
                               linestyle=(0, (3.0, 2.0)) if dashed else "solid", zorder=1))

    def arrow(points, color=edge, dashed=False, lw=1.5, head=8):
        if len(points) > 2:
            ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                    color=color, linewidth=lw * scale,
                    linestyle=(0, (3, 2)) if dashed else "solid", zorder=3)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=head * scale, color=color,
                                    linewidth=lw * scale,
                                    linestyle=(0, (3, 2)) if dashed else "solid",
                                    shrinkA=0, shrinkB=0, zorder=3))

    def down(x, y1, y2, **kwargs):
        arrow([(x, y1), (x, y2)], **kwargs)

    def panel(x, w, title):
        box(x, 8, w, 437, fill="#FCFDFE", lw=1.6)
        box(x, 8, w, 25, fill="#E8EDF3", lw=1.6)
        text(x + w / 2, 12, title, 18, align="center", weight=True,
             limit=(x + 5, 9, x + w - 5, 33))

    panel(8, 210, "Input & loop lowering")
    panel(232, 500, "Scout: constraint-guided schedule search")
    panel(748, 224, "Compilation & execution")

    # A model-graph operator and its exact, initialized loop representation.
    text(113, 47, "MatMul in a model graph", 16.5, align="center", weight=True)
    for x, y, w, h, value in [(25, 78, 24, 18, "A"), (25, 109, 24, 18, "B"),
                              (86, 87, 78, 31, "MatMul"), (183, 93, 24, 18, "C")]:
        box(x, y, w, h, fill="#F4F7FA")
        text(x + w / 2, y + (4 if value == "MatMul" else 1), value,
             17, align="center", font=mono)
    arrow([(49, 87), (65, 87), (86, 98)])
    arrow([(49, 118), (65, 118), (86, 107)])
    arrow([(164, 102), (183, 102)])
    text(113, 142, r"$C_{ij}=\sum_k A_{ik}B_{kj}$", 19, align="center")
    down(113, 167, 190)
    text(121, 170, "Lowering", 15)

    box(18, 190, 190, 130)
    text(26, 197, "Naive loop IR", 17, weight=True)
    naive = ["C[:,:] = 0", "for i in range(M):", " for j in range(N):",
             "  for k in range(K):", "   C[i,j] += (", "    A[i,k]*B[k,j])"]
    for index, line in enumerate(naive):
        text(24, 219 + index * 16, line, 15, font=mono, limit=(22, 216, 207, 319))

    box(18, 327, 190, 82, fill="#F4F6F8")
    text(26, 334, "Tensor shapes", 16, weight=True)
    text(26, 356, "A: M × K    B: K × N", 15.5)
    text(26, 374, "C: M × N", 15.5)
    text(26, 392, "Work: 2MNK FLOPs", 15.5)
    arrow([(208, 247), (225, 247), (225, 63), (244, 63)])

    # Schedule rules create a trace; CS/H act on generation and replay.
    box(244, 46, 476, 34, fill="#F1F4F7")
    text(482, 48, "Schedule space: split / reorder / cache / bind", 17,
         align="center", weight=True)
    text(482, 66, "Tiles, shared/local buffers, and block/thread bindings", 14.8, align="center")
    box(240, 91, 488, 287, fill="#F7FAFC", dashed=True)
    text(619, 96, "GA initialization + evolution", 16, align="center", weight=True)
    down(367, 80, 118)

    box(244, 118, 246, 35, fill=blue_fill, stroke=blue)
    text(367, 120, "1. CS + H: constraint search", 16.5, align="center", color=blue, weight=True)
    text(367, 139, "CS: sample feasible tiling factors", 15, align="center")
    down(367, 153, 161)
    box(244, 161, 246, 23, fill=blue_fill, stroke=blue)
    text(367, 165, "H: pre-replay trace check", 16, align="center", color=blue)
    down(367, 184, 192)

    box(244, 192, 246, 105)
    text(252, 196, "Replay → scheduled loop IR", 17, weight=True)
    scheduled = ["for by,bx in blocks:", "  for ty,tx in threads:",
                 "    c = 0", "    for ko,ki in tiles:", "      c+=As[...]*Bs[...]"]
    for index, line in enumerate(scheduled):
        text(252, 217 + index * 15.5, line, 15, font=mono, limit=(249, 214, 487, 297))
    down(367, 297, 305)
    box(244, 305, 246, 24, fill=blue_fill, stroke=blue)
    text(367, 309, "H: post-replay IR check", 16, align="center", color=blue)
    down(367, 329, 337)
    box(244, 337, 246, 33, fill="#EDF1F5")
    text(367, 338, "Postprocess + Verify", 16.5, align="center", weight=True)
    text(367, 356, "Original checks, including lowering", 14.8, align="center")

    # Draft is an analytic ranking signal, with a learned model after GA.
    box(518, 118, 202, 35, fill="#EFF3F7")
    text(619, 120, "Selection + mutation", 17, align="center", weight=True)
    text(619, 139, "Next generation", 15, align="center")
    arrow([(518, 135), (490, 135)])

    box(518, 192, 202, 83, fill=orange_fill, stroke=orange)
    text(619, 197, "2. Draft: coarse scoring", 16.5, align="center", color=orange, weight=True)
    text(619, 218, "IR features + target profile", 15, align="center")
    text(619, 237, r"$S(c)=\max(W/P,\ M/B)$", 17, align="center", color=orange)
    text(619, 257, "Analytic fitness for GA", 15, align="center")
    arrow([(490, 353), (504, 353), (504, 232), (518, 232)])
    arrow([(619, 192), (619, 153)])
    text(627, 168, "Fitness", 14.8)

    down(619, 275, 286)
    box(518, 286, 202, 51, fill="#FFF8EC", stroke=orange)
    text(619, 292, "Cross-generation archive", 15.5, align="center", weight=True)
    text(619, 313, "Rank by raw Draft scores", 15, align="center")
    arrow([(619, 337), (619, 385), (313, 385), (313, 398)], color=orange)
    text(627, 348, "After GA", 14.8, color=orange)

    box(244, 398, 138, 42, fill=orange_fill, stroke=orange)
    text(313, 401, "Archive shortlist", 16, align="center", weight=True, color=orange)
    text(313, 422, "verify_ratio", 15, align="center", font=mono)
    arrow([(382, 419), (410, 419)], color=orange)
    box(410, 398, 310, 42, fill=orange_fill, stroke=orange)
    text(565, 401, "Cost-model scoring and ranking", 17, align="center", weight=True, color=orange)
    text(565, 422, "Fine filter: select measurement candidates", 15, align="center")

    # A CUDA backend example, with validation retained before final codegen.
    arrow([(720, 419), (738, 419), (738, 81), (760, 81)])
    box(760, 46, 200, 70, fill="#EDF1F5")
    text(860, 52, "Backend lowering", 17, align="center", weight=True)
    text(860, 74, "Lower scheduled S-TIR", 15, align="center")
    text(860, 94, "Storage / sync / intrinsics", 15, align="center")
    down(860, 116, 133)
    box(760, 133, 200, 64)
    text(860, 138, "CUDA code generation", 17, align="center", weight=True)
    text(773, 160, "__global__ void", 15, font=mono)
    text(773, 179, "matmul(...) { ... }", 15, font=mono)
    down(860, 197, 214)
    box(760, 214, 200, 49, fill="#EDF1F5")
    text(860, 221, "CUDA compilation", 17, align="center", weight=True)
    text(860, 243, "PTX / cubin", 16, align="center")
    down(860, 263, 280)
    box(760, 280, 200, 56, fill="#F4F6F8")
    text(860, 285, "On-device measurement", 16, align="center", weight=True)
    text(860, 308, "Latency → tuning database", 15, align="center")
    down(860, 336, 375)
    text(875, 341, "Budget\nexhausted", 14.8)
    box(760, 375, 200, 45, fill="#DFE8F1", stroke="#435D77", lw=1.8)
    text(860, 380, "Best candidate binary", 17, align="center", weight=True)
    text(860, 402, "Deploy / execute", 15, align="center")

    # Measured performance feeds both model updates and subsequent batches.
    arrow([(960, 310), (978, 310), (978, 454), (236, 454), (236, 135), (244, 135)],
          color=feedback, dashed=True)
    arrow([(565, 454), (565, 440)], color=feedback, dashed=True)
    text(608, 463, "Measured latency: update the cost model and continue the next search batch", 15,
         color=feedback, align="center")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    for artist, limit in texts:
        extent = artist.get_window_extent(renderer)
        if not fig.bbox.contains(extent.x0, extent.y0) or not fig.bbox.contains(extent.x1, extent.y1):
            problems.append(f"Outside figure: {artist.get_text()}")
        if limit:
            left, top = ax.transData.transform((limit[0], limit[1]))
            right, bottom = ax.transData.transform((limit[2], limit[3]))
            if extent.x0 < left or extent.x1 > right or extent.y0 < bottom or extent.y1 > top:
                actual = ax.transData.inverted().transform(extent.get_points()).tolist()
                problems.append(f"Outside assigned text area: {artist.get_text()} ({actual}; font={artist.get_fontsize()})")
    if problems:
        raise ValueError("\n".join(problems))
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"architecture_matmul.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    manifest = {
        "status": "integrated", "format": "landscape two-column vector diagram",
        "size_inches": [printed_width / 72, height * scale / 72],
        "minimum_font_pt": 14.8 * scale,
        "code": "Illustrative pseudocode. Scheduled excerpt omits index reconstruction, loads, synchronization and writeback; As/Bs are shared-memory tiles and c is a local accumulator.",
        "method_semantics": "CS/H during candidate generation/replay; original postprocessing/Verify retained; Draft analytic fitness inside GA; cross-generation archive; generic cost-model scoring after shortlist; measurements update model and subsequent batches.",
        "manuscript_replaced": True, "visual_review": "pending",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("architecture_matmul.*")},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "size_inches": manifest["size_inches"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
