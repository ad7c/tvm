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
"""Draw an editable single-column architecture proposal from a MatMul example."""

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "papers/figures/architecture_matmul_v1"


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
    from matplotlib.patches import Rectangle, FancyArrowPatch

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": False})
    sans = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    bold = FontProperties(fname="C:/Windows/Fonts/msyhbd.ttc")
    mono = FontProperties(fname="C:/Windows/Fonts/consola.ttf")
    width, height = 222, 516
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    ink, edge = "#202D3B", "#647181"
    blue, blue_fill = "#245F91", "#E5F0F8"
    orange, orange_fill = "#A56519", "#FFF0D9"
    feedback = "#AB4545"
    text_artists = []

    def text(x, y, value, size=8, color=ink, align="left", font=None,
             rotation=0, weight=False, background=None):
        artist = ax.text(x, y, value, fontsize=size, color=color, ha=align, va="top",
                         fontproperties=font or (bold if weight else sans),
                         rotation=rotation, rotation_mode="anchor", linespacing=1.1,
                         bbox=dict(facecolor="white", edgecolor="none", pad=0.8) if background else None,
                         zorder=5)
        text_artists.append(artist)
        return artist

    def box(x, y, w, h, fill="white", stroke=edge, dashed=False, lw=0.65):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=stroke,
                               linewidth=lw, linestyle=(0, (2.3, 1.6)) if dashed else "solid",
                               zorder=1))

    def arrow(points, color=edge, dashed=False, lw=0.7, head=4):
        if len(points) > 2:
            ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                    color=color, linewidth=lw,
                    linestyle=(0, (2.5, 2)) if dashed else "solid", zorder=3)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=head, color=color, linewidth=lw,
                                    linestyle=(0, (2.5, 2)) if dashed else "solid",
                                    shrinkA=0, shrinkB=0, zorder=3))

    def down(y1, y2):
        arrow([(111, y1), (111, y2)])

    # Input graph and exact, unscheduled MatMul loop semantics.
    box(14, 4, 194, 100)
    box(14, 4, 194, 17, fill="#E8EDF3", stroke=edge)
    text(21, 7, "输入：矩阵乘 C = A × B", 8.7, weight=True)
    box(25, 28, 20, 12)
    box(25, 44, 20, 12)
    text(35, 29, "A", 8, align="center", font=mono)
    text(35, 45, "B", 8, align="center", font=mono)
    box(82, 33, 57, 19, fill="#F7F9FC")
    text(110.5, 36, "MatMul", 8, align="center", font=mono)
    box(175, 36, 20, 12)
    text(185, 37, "C", 8, align="center", font=mono)
    arrow([(45, 34), (64, 34), (82, 40)])
    arrow([(45, 50), (64, 50), (82, 45)])
    arrow([(139, 42.5), (175, 42.5)])
    naive = ["for i in range(M):", "  for j in range(N):", "    C[i,j] = 0",
             "    for k in range(K):", "      C[i,j] += A[i,k] * B[k,j]"]
    for index, line in enumerate(naive):
        text(22, 59 + index * 8.1, line, 7.25, font=mono)
    down(104, 112)

    box(14, 112, 194, 26, fill="#F4F6F8")
    text(111, 115, "调度规则：分块、重排、缓存、线程绑定", 7.9, align="center", weight=True)
    text(111, 126, "Split / Reorder / Cache / Bind", 7.1, align="center", font=mono)
    down(138, 148)

    # Search: H retains original validation; Draft operates on materialized IR.
    box(14, 148, 194, 251, fill="#FCFDFE", stroke="#465565", lw=0.85)
    text(21, 151, "Scout：约束搜索 + 分级筛选", 8.5, weight=True)
    box(22, 169, 179, 175, fill="#F7FAFC", dashed=True)
    text(111.5, 172, "GA 初始化与进化", 7.9, align="center", weight=True)

    box(39, 186, 148, 30, fill=blue_fill, stroke=blue)
    text(113, 189, "① 约束搜索：CS + H", 8.1, align="center", color=blue, weight=True)
    text(113, 202, "约束采样 · 回放前/后硬筛", 7.25, align="center")
    down(216, 224)

    box(39, 224, 148, 62, fill="white")
    text(44, 227, "调度后 IR（循环片段）", 7.6, weight=True)
    scheduled = ["for by,bx in grid(...):", "  for ty,tx in threads(...):",
                 "    for ko,ki in split(K):", "      c += As[...] * Bs[...]"]
    for index, line in enumerate(scheduled):
        text(43, 240 + index * 9.1, line, 7.25, font=mono)
    box(39, 286, 148, 12, fill="#EFF2F5")
    text(113, 288, "后处理与 Verify（含 lowering）", 6.9, align="center")
    down(298, 305)

    box(39, 305, 148, 27, fill=orange_fill, stroke=orange)
    text(113, 308, "② Draft 粗筛评分", 8.1, align="center", color=orange, weight=True)
    text(113, 320, "更新跨代归档", 7.2, align="center")
    arrow([(39, 318.5), (28, 318.5), (28, 201), (39, 201)], color=blue)
    text(26, 290, "选择父代 · 变异", 7.0, color=blue, rotation=90, background=True)
    down(332, 350)

    box(39, 350, 148, 17, fill="#FFF9EE", stroke=orange)
    text(113, 353, "GA 结束：归档 → 入围候选池", 7.3, align="center")
    down(367, 375)
    box(39, 375, 148, 20, fill=orange_fill, stroke=orange)
    text(113, 380, "代价模型打分与排序（精筛）", 7.7, color=orange, align="center", weight=True)
    down(395, 414)

    # The backend path is a CUDA example, not an instruction-level transcript.
    box(14, 414, 194, 31, fill="#EDF1F5")
    text(111, 418, "Lowering → 代码生成 → 编译", 8.2, align="center", weight=True)
    text(111, 431, "S-TIR → CUDA → PTX / cubin", 7.6, align="center")
    down(445, 453)
    box(14, 453, 194, 24, fill="#F5F7F9")
    text(111, 460, "硬件实测 · 更新性能数据库", 8.0, align="center", weight=True)
    arrow([(208, 465), (218, 465), (218, 201), (187, 201)], color=feedback, dashed=True)
    arrow([(218, 385), (187, 385)], color=feedback, dashed=True)
    text(208, 346, "实测反馈", 7.1, color=feedback, rotation=90, background=True)
    down(477, 490)
    text(119, 480, "预算结束", 6.9, color=edge)
    box(14, 490, 194, 20, fill="#DFE8F1", stroke="#435D77", lw=0.8)
    text(111, 494, "输出：最佳候选的目标二进制", 8.4, align="center", weight=True)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = []
    for artist in text_artists:
        extent = artist.get_window_extent(renderer)
        if extent.x0 < bounds.x0 or extent.x1 > bounds.x1 or extent.y0 < bounds.y0 or extent.y1 > bounds.y1:
            outside.append(artist.get_text())
    if outside:
        raise ValueError(f"Text outside figure: {outside}")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"architecture_matmul.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    manifest = {
        "status": "proposal", "format": "single-column vector diagram",
        "size_inches": [width / 72, height / 72],
        "code": "Illustrative pseudocode; scheduled excerpt omits initialization, index reconstruction, loads, synchronization and writeback.",
        "method_semantics": "CS/H during candidate generation/replay; original postprocessing/Verify retained; Draft in GA; cost model after shortlist; measurements update model and subsequent batches.",
        "manuscript_replaced": False, "visual_review": "pending",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("architecture_matmul.*")},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "size_inches": manifest["size_inches"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
