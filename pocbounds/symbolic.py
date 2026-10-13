"""Symbolic verification of tightness.

Solves the linear system that links the 81 response-function probabilities to
the observational and experimental distributions, and reports the general
solution.  `GaussianSolver` was named `GS` in the original script.
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import sympy as sp

# ---------- 外部已知符号（按你之前的定义） ----------
py0x0, py1x0, py0x1, py1x1, py0x2, py1x2, px0y0, px0y1, px0y2, px1y0, px1y1, px1y2, px2y0, px2y1 = sp.symbols(
    "py0x0 py1x0 py0x1 py1x1 py0x2 py1x2 px0y0 px0y1 px0y2 px1y0 px1y1 px1y2 px2y0 px2y1"
)

# ---------- 线性未知量：内部用 x1..x81 求解，但输出时全部替换为 p0000..p2222 ----------
x_syms = sp.symbols("x1:82")  # (x1, x2, ..., x81)

# 生成 p0000..p2222 的 81 个符号（3x3x3x3 的网格顺序）
p_names = []
for a in range(3):
    for b in range(3):
        for c in range(3):
            for d in range(3):
                p_names.append(f"p{a}{b}{c}{d}")
p_syms = sp.symbols(" ".join(p_names))  # (p0000, p0001, ..., p2222)

# x -> p 的符号替换字典（SymPy 符号到符号）
x_to_p = {x_syms[i]: p_syms[i] for i in range(81)}

# ---------- 你之前的 PNS 与下界函数 ----------
PNS = [0 for _ in range(81)]
PNS[15:18] = [1, 1, 1]

lb1 = 0


def lb2(k):
    acc = -k + 1
    acc += py0x0 + py1x1 + 1 - py0x2 - py1x2
    return acc

def lb3i(k, i):
    acc = -k + 1
    if i == 0:
        acc += py1x1 + 1 - py0x2 - py1x2 + px1y0 + px1y2 + px2y0 + px2y1 + px0y0
    elif i == 1:
        acc += py0x0 + 1 - py0x2 - py1x2 + px0y1 + px0y2 + px2y0 + px2y1 + px1y1
    elif i == 2:
        acc += py0x0 + py1x1 + px0y1 + px0y2 + px1y0 + px1y2 + 1 - px2y0 - px0y1
    return acc

def lb3(k):
    return [lb3i(k, i) for i in range(k)]

# ---------- 工具：把 linsolve 结果(集合)转为逐行 "p???? = 表达式" 文本 ----------

def solution_to_lines(sol_set):
    """
    sol_set: FiniteSet(tuple of 81 expressions in terms of x1..x81)
    返回：list[str]，长度 81，对应 p0000..p2222 的逐行表达式
    """
    if len(sol_set) == 0:
        return ["❌ 无解"]

    sol_tuple = list(sol_set)[0]  # 取唯一的参数化元组 (len=81)
    if len(sol_tuple) != 81:
        return [f"❌ 解元组长度异常：{len(sol_tuple)}，应为 81"]

    lines = []
    for i, expr in enumerate(sol_tuple):
        # 将表达式里的 x1..x81 替换为 p0000..p2222
        expr_p = sp.simplify(sp.sympify(expr).xreplace(x_to_p))
        lhs = p_syms[i]         # 左边符号 p????（作为 SymPy 符号）
        lines.append(f"{lhs} = {sp.sstr(expr_p)}")
    return lines

# ---------- 主类 ----------

class GaussianSolver:
    def __init__(self, equ_path, k, out_dir="solutions"):
        self.equ_list_A = []
        self.equ_list_b = []
        self.informer_list = []
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # 读取等式文件
        with open(equ_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.rstrip("\n")
                if not line:
                    continue
                out = line.split(',')
                mat = self.generate_matrix(out[0])
                cons = 1 if out[1] == "1" else sp.Symbol(out[1])
                self.equ_list_A.append(mat)
                self.equ_list_b.append(cons)

        self.generate_informer_list(k)

    def generate_matrix(self, index):
        # index 形如 "0x1x" 之类；维度 3x3x3x3，按 index 指示在子块中填 1
        matrix = np.zeros((3, 3, 3, 3), dtype=int)
        sl = [[0, 3], [0, 3], [0, 3], [0, 3]]
        for i, ch in enumerate(index):
            if ch != "x":
                v = int(ch)
                sl[i] = [v, v + 1]
        (a0, a1), (b0, b1), (c0, c1), (d0, d1) = sl
        matrix[a0:a1, b0:b1, c0:c1, d0:d1] = 1
        return matrix.flatten().tolist()

    def generate_informer_list(self, k):
        raw = [lb1, lb2(k)] + lb3(k)  # 共 1 + 1 + k = 5 个
        self.informer_list = [[PNS, expr] for expr in raw]

    def run_one(self, idx, informer_A, informer_B):
        A_rows = self.equ_list_A + [informer_A]
        b_rows = self.equ_list_b + [informer_B]
        A = sp.Matrix(A_rows)
        b = sp.Matrix(b_rows)

        rankA = A.rank()
        rankAb = A.row_join(b).rank()
        free_vars = A.shape[1] - rankA

        # 写文件名
        fname = os.path.join(self.out_dir, f"sol{idx+1}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write(f"[{idx+1}/{len(self.informer_list)}] Solution Report\n")
            f.write("="*80 + "\n")
            f.write(f"Matrix shape A: {A.shape}, b: {b.shape}\n")
            f.write(f"rank(A) = {rankA}, rank([A|b]) = {rankAb}\n")
            f.write(f"自由变量数 = {free_vars} (n={A.shape[1]})\n\n")

            if rankAb > rankA:
                f.write("❌ 系统不相容（无解）。\n")
                print(f"[Saved] {fname}  (无解)")
                return

            # 求解（以 x1..x81 为未知），随后统一替换为 p???? 符号并逐行输出
            sol_set = sp.linsolve((A, b), x_syms)
            lines = solution_to_lines(sol_set)

            f.write("逐行解（已将 x1..x81 替换为 p0000..p2222）:\n")
            for line in lines:
                f.write(line + "\n")

        print(f"[Saved] {fname}")

    def run(self):
        for i, (ia, ib) in enumerate(self.informer_list):
            self.run_one(i, ia, ib)

# ---------- 入口 ----------


GS = GaussianSolver  # backwards-compatible alias
