# -*- coding: utf-8 -*-
"""绘制系统整体架构流程框图（§3 方法的主图）。"""
import os
import sys
from pathlib import Path

# 与 plot_paper_pilot_20261008.py 一致：绘图进程需要自身的 DLL 目录 + 串行 MKL。
library_bin = Path(sys.executable).parent / "Library/bin"
if os.name == "nt" and library_bin.is_dir():
    os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(11, 3.4), dpi=200)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

FILL = '#F3F6F9'
EDGE = '#365A78'
TXT = '#20262D'
FB = '#7D8792'


def box(lines, x, y, w, h, fill=FILL, edge=EDGE):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006",
                                linewidth=1.5, edgecolor=edge, facecolor=fill))
    ax.text(x, y, "\n".join(lines), ha='center', va='center',
            fontsize=10.5, color=TXT, linespacing=1.45)


def arrow(x1, x2, y, color=TXT, lw=1.8):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color))


boxes = [
    (["剪枝", "(CS + H)"], 0.09),
    (["初始化", "合法候选种群"], 0.27),
    (["GA 迭代", "粗筛 · 解析评分", "指导变异方向"], 0.47),
    (["精筛", "代价模型 XGBoost"], 0.69),
    (["上机测试", "Builder + Runner"], 0.91),
]

W, H, Y = 0.155, 0.42, 0.66
for lines, x in boxes:
    box(lines, x, Y, W, H)

for i in range(len(boxes) - 1):
    arrow(boxes[i][1] + W / 2, boxes[i + 1][1] - W / 2, Y)

# 反馈回路：上机测试 → 更新代价模型（回到精筛）
x_last = boxes[-1][1]
x_pre = boxes[-2][1]
yb = Y - H / 2 - 0.05
ax.annotate("", xy=(x_pre, yb), xytext=(x_last, yb),
            arrowprops=dict(arrowstyle='-|>', lw=1.6, color=FB,
                            connectionstyle="arc3,rad=-0.35"))
ax.text((x_last + x_pre) / 2, yb - 0.17, "实测反馈 · 更新代价模型",
        ha='center', va='center', fontsize=9.5, color=FB)

plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
out = r"F:\tvm\papers\figures\pilot_20261008\system_architecture"
plt.savefig(out + ".png", bbox_inches='tight', facecolor='white')
plt.savefig(out + ".svg", bbox_inches='tight', facecolor='white')
print("saved:", out + ".png")
print("saved:", out + ".svg")
