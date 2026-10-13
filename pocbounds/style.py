"""Matplotlib styling used for the figures in the paper."""

import matplotlib as mpl


def set_times_paper_style():
    mpl.rcParams.update({
        # ---------------- Font family ----------------
        # 对应 LaTeX 的 \usepackage{times}
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",  # Windows / macOS 常见
            "Times",            # 通用
            "Nimbus Roman",     # Linux 常见（等价 Times）
            "DejaVu Serif"      # fallback
        ],

        # 数学字体：最接近 Times 的数学外观
        "mathtext.fontset": "stix",
        # 如果你希望数学也完全不斜（可选）
        # "mathtext.default": "regular",

        # ---------------- Font sizes ----------------
        "font.size": 10,          # 全局默认 = 10
        # "axes.titlesize": 10,     # 图标题 = 12
        "axes.labelsize": 8,     # 坐标轴 = 10
        "xtick.labelsize": 8,    # 刻度 = 10
        "ytick.labelsize": 8,
        "legend.fontsize": 8,    # 图例 = 10
        # "axes.titleweight": "bold",

        # ---------------- Clean look ----------------
        "axes.grid": False,       # 无网格
        "axes.linewidth": 0.8,
        "lines.linewidth": 0.6,
        "lines.markersize": 2.5,

        # ---------------- Export quality ----------------
        "savefig.dpi": 600,       # PNG 高清；PDF 本身矢量
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
