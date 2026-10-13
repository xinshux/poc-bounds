"""Reproduces the bound-tightening comparison in Section 5 of the paper."""

from pocbounds import ConjecturePNS, set_times_paper_style
from pocbounds.recursive import RecursivePNS
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import random


class TightnessExperiment:
    # n k yi xj n_sample
    def __init__(self, n_dimension=20, k=20, n_sample=100):
        self.n_sample = n_sample
        self.n_dimension = n_dimension
        self.k = k
        self.yi = [i+1 for i in range(k)]
        self.xj = [i+1 for i in range(k)]

        # 用实例属性（避免类属性在多个实例间“串味”）
        self.lb, self.ub = [], []
        self.lbx, self.ubx = [], []
        self.lby, self.uby = [], []
        self.lbxy, self.ubxy = [], []

        self.rlb, self.rub = [], []
        self.rlbx, self.rubx = [], []
        self.rlby, self.ruby = [], []
        self.rlbxy, self.rubxy = [], []

        for dimension in range(self.k, self.n_dimension + 1):
            probs = self.sample(current_dimension=dimension, n_sample=n_sample)
            lbs, ubs, lbxs, ubxs, lbys, ubys, lbxys, ubxys = [], [], [], [], [], [], [], []
            rlbs, rubs, rlbxs, rubxs, rlbys, rubys, rlbxys, rubxys = [], [], [], [], [], [], [], []
            for prob in probs:
                # 这里给 ConjecturePNS 传对参数：n=当前维度；k=当前维度（若你的 k 定义为“选对角的个数”，也可改为 k=min(3, dimension)）
                # p/q 是 x_p, y_q 的索引，演示固定为 1；你也可以随维度变动或外部传参
                my_PNS = ConjecturePNS(n=dimension, k=self.k, p=1, q=1, prob=prob,yi=self.yi,xj=self.xj)
                re_PNS = RecursivePNS(n=dimension,m=dimension,k=self.k,p=1,q=1,prob=prob,yi=self.yi,xj=self.xj)

                #recursion_PNS = RecursivePNS(n=dimension, k=dimension, p=1, q=1, prob=prob)
                # 此处还有一堆

                lbs.append(my_PNS.lb)
                ubs.append(my_PNS.ub)
                lbxs.append(my_PNS.lbx)
                ubxs.append(my_PNS.ubx)
                lbys.append(my_PNS.lby)
                ubys.append(my_PNS.uby)
                lbxys.append(my_PNS.lbxy)
                ubxys.append(my_PNS.ubxy)

                rlbs.append(re_PNS.lb)
                rubs.append(re_PNS.ub)
                rlbxs.append(re_PNS.lbx)
                rubxs.append(re_PNS.ubx)
                rlbys.append(re_PNS.lby)
                rubys.append(re_PNS.uby)
                rlbxys.append(re_PNS.lbxy)
                rubxys.append(re_PNS.ubxy)
                # if my_PNS.lbx < re_PNS.lbx:
                #     print(f"----------------error in lbx:----------------\n prob:{prob} \n myPNS: {my_PNS.lbx},rePNS: {re_PNS.lbx}")
                #     exit(1)
                # if my_PNS.ubx > re_PNS.ubx:
                #     print(
                #         f"----------------error in ubx:----------------\n prob:{prob} \n myPNS: {my_PNS.ubx},rePNS: {re_PNS.ubx}")
                #     exit(1)
                # if my_PNS.lbxy < re_PNS.lbxy:
                #     print(
                #         f"----------------error in lbxy:----------------\n prob:{prob} \n myPNS: {my_PNS.lbxy},rePNS: {re_PNS.lbxy}")
                #     exit(1)
                # if my_PNS.ubxy > re_PNS.ubxy:
                #     print(
                #         f"----------------error in ubxy:----------------\n prob:{prob} \n myPNS: {my_PNS.ubxy},rePNS: {re_PNS.ubxy}")
                #     exit(1)

            self.lb.append(np.mean(lbs))
            self.ub.append(np.mean(ubs))
            self.lbx.append(np.mean(lbxs))
            self.ubx.append(np.mean(ubxs))
            self.lby.append(np.mean(lbys))
            self.uby.append(np.mean(ubys))
            self.lbxy.append(np.mean(lbxys))
            self.ubxy.append(np.mean(ubxys))

            self.rlb.append(np.mean(rlbs))
            self.rub.append(np.mean(rubs))
            self.rlbx.append(np.mean(rlbxs))
            self.rubx.append(np.mean(rubxs))
            self.rlby.append(np.mean(rlbys))
            self.ruby.append(np.mean(rubys))
            self.rlbxy.append(np.mean(rlbxys))
            self.rubxy.append(np.mean(rubxys))
        # print(self.lby, self.uby)


    def sample(self, current_dimension, n_sample=100):
        """
        返回长度为 n_sample 的列表，每个元素是一个概率分布字典：
          - P(xi,yj) 观测联合（n×n，总和=1）
          - P(yj_xi) 实验/反事实（每个 i 的一行和=1）
        且满足：P(xi,yj) <= P(yj_xi) <= 1 - P(xi,not yj)
        """
        n = int(current_dimension)
        if n < 2:
            raise ValueError("current_dimension 必须 >= 2")
        if n_sample < 1:
            raise ValueError("n_sample 必须为正整数")

        probs = []
        for _ in range(n_sample):
            # 1) 生成观测联合分布 P(x,y)（和为 1）
            Pxy = np.random.dirichlet(np.ones(n * n)).reshape(n, n)  # (n,n)

            # 2) 为每个 x_i 生成满足约束的 P(y|do(x_i))
            Q = np.zeros((n, n))
            for i in range(n):
                lb = Pxy[i, :].copy()  # 下界：P(xi,yj)
                px_i = lb.sum()  # P(X=xi)
                R = 1.0 - px_i  # 剩余可分配量
                if R <= 1e-12:
                    Q[i, :] = lb  # 几乎满了，直接取下界
                else:
                    w = np.random.dirichlet(np.ones(n))  # 把 R 随机分配到 n 个 y
                    Q[i, :] = lb + R * w  # 自然满足上界：ub = lb + R

                # （可选）数值断言
                # ub = 1.0 - px_i + lb
                # assert np.all(Q[i, :] + 1e-9 >= lb) and np.all(Q[i, :] <= ub + 1e-9)
                # assert np.isclose(Q[i, :].sum(), 1.0, atol=1e-9)

            # 3) 组装一个 2*n^2 的字典
            prob = {}
            for i in range(n):
                for j in range(n):
                    prob[f"P(x{i + 1},y{j + 1})"] = float(Pxy[i, j])
                    prob[f"P(y{j + 1}_x{i + 1})"] = float(Q[i, j])
            probs.append(prob)

        return probs

    def sample_conf(self, current_dimension, n_sample=100, n_y=None, n_z=None, eps=1e-12):
        """
        使用 Confounder 结构生成 prob 列表：
          Z -> X, Z -> Y, X -> Y

        约定（按你要求）：
          - 默认 n_x = current_dimension
          - 默认 n_y = current_dimension
          - 默认 n_z = current_dimension

        返回：
          probs: list[dict]
            每个 dict 含：
              - P(xi,yj) 观测联合分布
              - P(yj_xi) 实验分布（这里等价于 do(X=xi) 下的 Y 分布）
        """
        n_x = int(current_dimension)
        if n_x < 2:
            raise ValueError("current_dimension 必须 >= 2")
        if n_sample < 1:
            raise ValueError("n_sample 必须为正整数")

        if n_y is None:
            n_y = n_x
        if n_z is None:
            n_z = n_x  # 你要求 n_z 也等于 current_dimension

        n_y = int(n_y)
        n_z = int(n_z)
        if n_y < 2 or n_z < 2:
            raise ValueError("n_y 和 n_z 都必须 >= 2")

        probs = []

        for _ in range(n_sample):
            # 1) p(z)
            p_z = np.random.dirichlet(np.ones(n_z))  # (n_z,)

            # 2) p(x | z)
            p_x_given_z = np.zeros((n_z, n_x))
            for k in range(n_z):
                p_x_given_z[k] = np.random.dirichlet(np.ones(n_x))

            # 3) p(y | x, z)
            p_y_given_xz = np.zeros((n_x, n_z, n_y))
            for i in range(n_x):
                for k in range(n_z):
                    p_y_given_xz[i, k] = np.random.dirichlet(np.ones(n_y))

            # 4) 生成 P(x,y) 与 Q(y|do(x))  (你代码里记为 P(y_j_x_i))
            Pxy = np.zeros((n_x, n_y))
            Q = np.zeros((n_x, n_y))

            for i in range(n_x):
                for j in range(n_y):
                    tmp_xy = 0.0
                    tmp_yx = 0.0
                    for k in range(n_z):
                        tmp_xy += p_z[k] * p_x_given_z[k, i] * p_y_given_xz[i, k, j]
                        tmp_yx += p_z[k] * p_y_given_xz[i, k, j]
                    Pxy[i, j] = tmp_xy
                    Q[i, j] = tmp_yx

            # ---- 数值校验（建议保留）----
            s_xy = Pxy.sum()
            if not np.isfinite(s_xy) or abs(s_xy - 1.0) > 1e-8:
                # 理论上应为 1，这里做一次保险归一化
                if s_xy < eps:
                    raise RuntimeError("Pxy sum 太小，出现数值问题")
                Pxy = Pxy / s_xy

            row_sums = Q.sum(axis=1)
            if np.any(~np.isfinite(row_sums)):
                raise RuntimeError("Q row sums 出现非有限值")
            # 理论上每行都应为 1；若有轻微误差，归一化到 1
            for i in range(n_x):
                if abs(row_sums[i] - 1.0) > 1e-8:
                    if row_sums[i] < eps:
                        raise RuntimeError("Q row sum 太小，出现数值问题")
                    Q[i, :] = Q[i, :] / row_sums[i]

            # 5) 打包成你现有 ConjecturePNS / RecursivePNS 需要的 prob dict
            prob = {}
            for i in range(n_x):
                for j in range(n_y):
                    prob[f"P(x{i + 1},y{j + 1})"] = float(Pxy[i, j])
                    prob[f"P(y{j + 1}_x{i + 1})"] = float(Q[i, j])

            probs.append(prob)

        return probs

    def draw_dim_d(self, dimension, diff=None):
        """
        画指定 dimension 的逐-sample 图（并按 rec_ub 从小到大排序后再画）：
          图1：Rec / Conj bounds
          图2：difference (ΔLB, ΔUB)

        并在每次计算完成后，额外输出 5 个统计量 + 一段可直接复制到论文的 LaTeX 表格代码。

        记号对应（按你的定义）：
          - [a_i, b_i] = Thm2 / conjecture bounds  -> a_i=conj_lb, b_i=conj_ub
          - [c_i, d_i] = Li&Pearl / recursive bounds -> c_i=rec_lb, d_i=rec_ub

        diff 参数：
          - diff is None or diff <= 0 : 不过滤 sample
          - diff > 0 : 仅保留 gain > diff 的 sample
            gain = max( max(conj_lb - rec_lb, 0), max(rec_ub - conj_ub, 0) )
        """
        import matplotlib.ticker as mticker
        import numpy as np
        import matplotlib.pyplot as plt

        # ------------------ 参数检查 ------------------
        if not isinstance(dimension, int):
            raise TypeError("dimension 必须是 int")
        if dimension < 2:
            raise ValueError("dimension 必须 >= 2")
        if dimension < self.k:
            raise ValueError(f"dimension={dimension} 不能小于 k={self.k}")
        if dimension > self.n_dimension:
            raise ValueError(f"dimension={dimension} 超过 n_dimension={self.n_dimension}")

        n_sample = int(self.n_sample)

        # 你当前默认用 sample()；如果想换 confounder 结构，改这一行即可
        probs = self.sample(current_dimension=dimension, n_sample=n_sample)
        # probs = self.sample_conf(current_dimension=dimension, n_sample=n_sample)

        conj_lb, conj_ub = [], []
        rec_lb, rec_ub = [], []

        yi = self.yi
        xj = self.xj

        # ------------------ 1) 计算所有 sample 的 bounds ------------------
        for prob in probs:
            my_PNS = ConjecturePNS(
                n=dimension, k=self.k, p=1, q=1,
                prob=prob, yi=yi, xj=xj
            )
            re_PNS = RecursivePNS(
                n=dimension, m=dimension, k=self.k, p=1, q=1,
                prob=prob, yi=yi, xj=xj
            )

            conj_lb.append(my_PNS.lb)
            conj_ub.append(my_PNS.ub)
            rec_lb.append(re_PNS.lb)
            rec_ub.append(re_PNS.ub)

        conj_lb = np.array(conj_lb, dtype=float)
        conj_ub = np.array(conj_ub, dtype=float)
        rec_lb = np.array(rec_lb, dtype=float)
        rec_ub = np.array(rec_ub, dtype=float)

        # ------------------ 2) diff 过滤 ------------------
        if diff is not None and diff > 0:
            imp_lb = conj_lb - rec_lb  # LB 提升：越大越好
            imp_ub = rec_ub - conj_ub  # UB 收紧：越大越好
            gain = np.maximum(np.maximum(imp_lb, 0.0), np.maximum(imp_ub, 0.0))
            mask = gain > diff

            conj_lb = conj_lb[mask]
            conj_ub = conj_ub[mask]
            rec_lb = rec_lb[mask]
            rec_ub = rec_ub[mask]

            if len(conj_lb) == 0:
                print(f"[draw_dim_d] No samples satisfy diff > {diff} (dimension={dimension})")
                return

        # ------------------ 3) 按 rec_ub 从小到大排序 ------------------
        order = np.argsort(rec_ub, kind="mergesort")  # 稳定排序
        conj_lb = conj_lb[order]
        conj_ub = conj_ub[order]
        rec_lb = rec_lb[order]
        rec_ub = rec_ub[order]

        # x 轴：排序后的 rank
        x = np.arange(1, len(conj_lb) + 1)

        # ------------------ 4) 计算 5 个统计量 + 输出 LaTeX 表格 ------------------
        # 按你定义：a,b 是 Thm2；c,d 是 Li&Pearl
        a = conj_lb
        b = conj_ub
        c = rec_lb
        d = rec_ub

        N_used = len(a)  # 过滤后用于画图/统计的样本数
        N_total = int(self.n_sample)  # 原始样本数（例如 10000）

        # “/10000” 的口径严格对应 N_total；如果 diff>0，N_used < N_total，会稀释平均值。
        # 这里同时给出 “按 N_total 归一” 和 “按 N_used 归一”（供你核对/选择）。
        inc_lb_sum = float(np.sum(a - c))  # Σ(a_i - c_i)
        dec_ub_sum = float(np.sum(d - b))  # Σ(d_i - b_i)
        gap_li_sum = float(np.sum(d - c))  # Σ(d_i - c_i)
        gap_thm2_sum = float(np.sum(b - a))  # Σ(b_i - a_i)

        avg_inc_lb_total = inc_lb_sum / N_total
        avg_dec_ub_total = dec_ub_sum / N_total
        avg_gap_li_total = gap_li_sum / N_total
        avg_gap_thm2_total = gap_thm2_sum / N_total

        avg_inc_lb_used = inc_lb_sum / max(N_used, 1)
        avg_dec_ub_used = dec_ub_sum / max(N_used, 1)
        avg_gap_li_used = gap_li_sum / max(N_used, 1)
        avg_gap_thm2_used = gap_thm2_sum / max(N_used, 1)

        narrower_count = int(np.sum((a > c) | (b < d)))  # Σ f_i

        print(f"[Stats] dimension={dimension}, diff={diff}, N_used={N_used}, N_total={N_total}")
        print(f"  Avg increased LB   (Σ(a-c)/N_total): {avg_inc_lb_total:.6f}   (per used: {avg_inc_lb_used:.6f})")
        print(f"  Avg decreased UB   (Σ(d-b)/N_total): {avg_dec_ub_total:.6f}   (per used: {avg_dec_ub_used:.6f})")
        print(f"  Avg gap Li&Pearl   (Σ(d-c)/N_total): {avg_gap_li_total:.6f}   (per used: {avg_gap_li_used:.6f})")
        print(f"  Avg gap Theorem 2  (Σ(b-a)/N_total): {avg_gap_thm2_total:.6f}   (per used: {avg_gap_thm2_used:.6f})")
        print(f"  # narrower samples (a>c or b<d): {narrower_count}")


        # ------------------ 5) 画图参数 ------------------
        TITLE_FT, LABEL_FT, LEGEND_FT = 50, 50, 35
        title_suffix_plot = f", diff>{diff}" if diff is not None and diff > 0 else ""

        # ------------------ 图1：bounds ------------------
        plt.figure(figsize=(14, 7))

        plt.plot(x, rec_lb, linestyle='-', label='Li and Pearl-LB')
        plt.plot(x, rec_ub, linestyle='-', label='Li and Pearl-UB')

        plt.plot(x, conj_lb, linestyle='-', label='Thm2-LB')
        plt.plot(x, conj_ub, linestyle='-', label='Thm2-UB')

        ax = plt.gca()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
        ax.margins(x=0.01, y=0.05)

        plt.title(f'PNS(k) Bounds (n={dimension}{title_suffix_plot})', fontsize=TITLE_FT)
        plt.xlabel(f'Valid samples/{self.n_sample}', fontsize=LABEL_FT)
        plt.ylabel('Value', fontsize=LABEL_FT)
        plt.legend(fontsize=LEGEND_FT)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # ------------------ 图2：difference ------------------
        d_lb = conj_lb - rec_lb  # ΔLB = Thm2 - Li&Pearl
        d_ub = rec_ub - conj_ub  # ΔUB = Li&Pearl - Thm2

        plt.figure(figsize=(14, 7))
        plt.plot(x, d_lb, linestyle='-', label='ΔLB = LB_Thm2 − LB_Li and Pearl')
        plt.plot(x, d_ub, linestyle='-', label='ΔUB = UB_Li and Pearl − UB_Thm2')

        ax2 = plt.gca()
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
        ax2.margins(x=0.01, y=0.05)
        plt.axhline(0, linewidth=1, linestyle=':', color='gray')

        plt.title(f'DIFF per Sample (n={dimension}{title_suffix_plot})', fontsize=TITLE_FT)
        plt.xlabel(f'Valid samples/{self.n_sample}', fontsize=LABEL_FT)
        plt.ylabel('Difference', fontsize=LABEL_FT)
        plt.legend(fontsize=LEGEND_FT)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.show()

    def draw_dim_top(self, dimension, diff=None, top_k=100, seed=0):
        """
        画指定 dimension 的逐-sample 图（先选 difference 最大的 top_k 个 sample，
        然后按 rec_ub 从小到大排序后再画）：

          图1：Rec / Thm2 bounds
          图2：difference (ΔLB, ΔUB)

        并在每次计算完成后，额外输出 5 个统计量。

        记号对应（按你的定义）：
          - [a_i, b_i] = Thm2 bounds  -> a_i=conj_lb, b_i=conj_ub
          - [c_i, d_i] = Li&Pearl bounds -> c_i=rec_lb, d_i=rec_ub

        difference/gain 定义（与你之前一致）：
          imp_lb = conj_lb - rec_lb      (LB 提升)
          imp_ub = rec_ub - conj_ub      (UB 收紧)
          gain   = max(max(imp_lb,0), max(imp_ub,0))

        diff 参数：
          - 保留兼容：如果 top_k 为 None 且 diff>0，才会用阈值过滤 gain>diff
          - 推荐：使用 top_k=100，不用 diff 一刀切

        Parameters
        ----------
        dimension : int
        diff : float or None
        top_k : int or None
            选 difference 最大的 top_k 个 sample；None 表示不做 top-k 选择（可配合 diff）
        seed : int
            用于稳定 tie-breaking（当 gain 一样时）
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        set_times_paper_style()

        # ------------------ 参数检查 ------------------
        if not isinstance(dimension, int):
            raise TypeError("dimension 必须是 int")
        if dimension < 2:
            raise ValueError("dimension 必须 >= 2")
        if dimension < self.k:
            raise ValueError(f"dimension={dimension} 不能小于 k={self.k}")
        if dimension > self.n_dimension:
            raise ValueError(f"dimension={dimension} 超过 n_dimension={self.n_dimension}")

        n_sample = int(self.n_sample)
        rng = np.random.default_rng(seed)

        # 你当前默认用 sample()；如果想换 confounder 结构，改这一行即可
        probs = self.sample(current_dimension=dimension, n_sample=n_sample)
        # probs = self.sample_conf(current_dimension=dimension, n_sample=n_sample)

        conj_lb, conj_ub, rec_lb, rec_ub = [], [], [], []
        yi = self.yi
        xj = self.xj

        # ------------------ 1) 计算所有 sample 的 bounds ------------------
        for prob in probs:
            my_PNS = ConjecturePNS(
                n=dimension, k=self.k, p=1, q=1,
                prob=prob, yi=yi, xj=xj
            )
            re_PNS = RecursivePNS(
                n=dimension, m=dimension, k=self.k, p=1, q=1,
                prob=prob, yi=yi, xj=xj
            )

            conj_lb.append(my_PNS.lb)
            conj_ub.append(my_PNS.ub)
            rec_lb.append(re_PNS.lb)
            rec_ub.append(re_PNS.ub)

        conj_lb = np.asarray(conj_lb, dtype=float)
        conj_ub = np.asarray(conj_ub, dtype=float)
        rec_lb = np.asarray(rec_lb, dtype=float)
        rec_ub = np.asarray(rec_ub, dtype=float)

        # ------------------ 2) 计算 difference 强度 gain ------------------
        imp_lb = conj_lb - rec_lb  # LB 提升：越大越好
        imp_ub = rec_ub - conj_ub  # UB 收紧：越大越好
        gain = np.maximum(np.maximum(imp_lb, 0.0), np.maximum(imp_ub, 0.0))

        # ------------------ 3) 选择 difference 最大的 top_k 个 sample ------------------
        if top_k is not None and top_k > 0 and len(gain) > top_k:
            # 为了在 gain 相同的情况下结果稳定：加一个极小的随机扰动做 tie-break
            jitter = rng.random(len(gain)) * 1e-12
            idx_sorted = np.argsort(gain + jitter, kind="mergesort")  # 从小到大
            idx_top = idx_sorted[-top_k:]  # 取最大的 top_k
            conj_lb = conj_lb[idx_top]
            conj_ub = conj_ub[idx_top]
            rec_lb = rec_lb[idx_top]
            rec_ub = rec_ub[idx_top]
            gain = gain[idx_top]
        elif top_k is None and diff is not None and diff > 0:
            # 兼容旧逻辑：仅当不做 top-k 时才启用阈值过滤
            mask = gain > diff
            conj_lb = conj_lb[mask]
            conj_ub = conj_ub[mask]
            rec_lb = rec_lb[mask]
            rec_ub = rec_ub[mask]
            gain = gain[mask]
            if len(gain) == 0:
                print(f"[draw_dim_d] No samples satisfy gain > {diff} (dimension={dimension})")
                return

        print(
            f"[draw_dim_d] dimension={dimension}, selected={len(gain)}/{n_sample}, "
            f"gain range=[{gain.min():.6f}, {gain.max():.6f}]"
        )

        # ------------------ 4) 按 rec_ub 从小到大排序（你的要求） ------------------
        order = np.argsort(rec_ub, kind="mergesort")
        conj_lb = conj_lb[order]
        conj_ub = conj_ub[order]
        rec_lb = rec_lb[order]
        rec_ub = rec_ub[order]
        gain = gain[order]

        x = np.arange(1, len(conj_lb) + 1)

        # ------------------ 5) 计算 5 个统计量 ------------------
        # a,b 是 Thm2；c,d 是 Li&Pearl
        a, b, c, d = conj_lb, conj_ub, rec_lb, rec_ub
        N_used = len(a)
        N_total = int(self.n_sample)

        inc_lb_sum = float(np.sum(a - c))  # Σ(a_i - c_i)
        dec_ub_sum = float(np.sum(d - b))  # Σ(d_i - b_i)
        gap_li_sum = float(np.sum(d - c))  # Σ(d_i - c_i)
        gap_thm2_sum = float(np.sum(b - a))  # Σ(b_i - a_i)

        # 注意：top-k 后 N_used < N_total
        # 若你要“平均到全体 10000 分布”的口径，用 /N_total（会被 top-k 稀释）
        # 若你要“top-k 子集的平均改善”，用 /N_used（更符合 top-k 图的语义）
        avg_inc_lb_total = inc_lb_sum / N_total
        avg_dec_ub_total = dec_ub_sum / N_total
        avg_gap_li_total = gap_li_sum / N_total
        avg_gap_thm2_total = gap_thm2_sum / N_total

        avg_inc_lb_used = inc_lb_sum / max(N_used, 1)
        avg_dec_ub_used = dec_ub_sum / max(N_used, 1)
        avg_gap_li_used = gap_li_sum / max(N_used, 1)
        avg_gap_thm2_used = gap_thm2_sum / max(N_used, 1)

        narrower_count = int(np.sum((a > c) | (b < d)))  # a>c 或 b<d

        print(f"[Stats] dimension={dimension}, top_k={top_k}, diff={diff}, N_used={N_used}, N_total={N_total}")
        print(f"  Avg increased LB   (Σ(a-c)/N_total): {avg_inc_lb_total:.6f}   (per used: {avg_inc_lb_used:.6f})")
        print(f"  Avg decreased UB   (Σ(d-b)/N_total): {avg_dec_ub_total:.6f}   (per used: {avg_dec_ub_used:.6f})")
        print(f"  Avg gap Li&Pearl   (Σ(d-c)/N_total): {avg_gap_li_total:.6f}   (per used: {avg_gap_li_used:.6f})")
        print(f"  Avg gap Theorem 2  (Σ(b-a)/N_total): {avg_gap_thm2_total:.6f}   (per used: {avg_gap_thm2_used:.6f})")
        print(f"  # narrower samples (a>c or b<d): {narrower_count}")

        # ------------------ 6) 画图参数（论文友好；你也可以改回大字号） ------------------
        # TITLE_FT, LABEL_FT, LEGEND_FT, TICK_FT = 10, 9, 8, 8
        title_suffix_plot = f", top-{min(top_k, N_used)}" if top_k is not None else ""
        if top_k is None and diff is not None and diff > 0:
            title_suffix_plot += f", gain>{diff}"

        # ------------------ 图1：bounds ------------------
        plt.figure(figsize=(3.25, 2.4))

        plt.plot(x, rec_ub, linestyle='-', color='black', label='Li and Pearl-UB')
        plt.plot(x, conj_ub, linestyle='-', color='green', label='Thm2-UB')
        plt.plot(x, conj_lb, linestyle='-', color='red', label='Thm2-LB')
        plt.plot(x, rec_lb, linestyle='-', color='blue', label='Li and Pearl-LB')

        ax = plt.gca()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
        ax.margins(x=0.01, y=0.05)
        k=self.k
        # plt.title(f'PNS({k}) Bounds (n={dimension})')
        plt.xlabel('Samples')
        plt.ylabel('Bounds for PNS')
        plt.xticks()
        plt.yticks()
        # plt.legend()
        plt.legend(loc="best",frameon=True)
        # plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"bounds_top{top_k}_n{dimension}.pdf", bbox_inches="tight")  # 不糊（矢量）

        # ------------------ 图2：difference ------------------
        d_lb = conj_lb - rec_lb  # ΔLB = Thm2 - Li&Pearl
        d_ub = rec_ub - conj_ub  # ΔUB = Li&Pearl - Thm2

        plt.figure(figsize=(3.25, 2.4))
        plt.plot(x, d_ub, linestyle='-', color='black',label=r'$\Delta UB = UB_{\mathrm{Li&Pearl}} - UB_{\mathrm{Thm2}}$')
        plt.plot(x, d_lb, linestyle='-', color='green', label=r'$\Delta LB = LB_{\mathrm{Thm2}} - LB_{\mathrm{Li&Pearl}}$')

        ax2 = plt.gca()
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
        ax2.margins(x=0.01, y=0.05)
        plt.axhline(0, linewidth=1, linestyle=':', color='gray')

        # plt.title(f'DIFF per Sample (n={dimension})')
        plt.xlabel('Samples')
        plt.ylabel('Bounds Differences for PNS')
        plt.xticks()
        plt.yticks()
        # plt.legend()
        plt.legend(loc="best",frameon=True)
        # plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"bounds_top{top_k}_n{dimension}_diff.pdf", bbox_inches="tight")  # 不糊（矢量）

        plt.show()
        
    def draw_lb_values(self,
                    dimension,
                    n_sample=None,
                    use_conf=False,
                    sort_by="diff_abs",
                    top_k=None,
                    seed=0,
                    show_diff=True,
                    save_prefix=None):
        # 我只想看两个method的lower bd在不同维度下的区别
        """
        画指定 dimension 下，每个 sample 的两种方法 lower bound 具体值（不取平均）。

        Parameters
        ----------
        dimension : int
            当前要看的维度 n
        n_sample : int or None
            这次临时采样多少个数据；None 表示用 self.n_sample
        use_conf : bool
            False -> 用 self.sample()
            True  -> 用 self.sample_conf()
        sort_by : str
            排序方式，可选：
            - "diff_abs" : 按 |LB_Thm2 - LB_LiPearl| 从小到大排序
            - "diff"     : 按 (LB_Thm2 - LB_LiPearl) 从小到大排序
            - "rec_lb"   : 按 Li&Pearl-LB 从小到大排序
            - "conj_lb"  : 按 Thm2-LB 从小到大排序
            - "none"     : 不排序，保持原始 sample 顺序
        top_k : int or None
            如果不是 None，只保留差异最大的 top_k 个 sample
        seed : int
            tie-break / 复现实验用
        show_diff : bool
            是否额外画一张 difference 图
        save_prefix : str or None
            若给定，如 "lb_compare"，则保存为：
            lb_compare_n{dimension}.pdf
            lb_compare_n{dimension}_diff.pdf

        Returns
        -------
        result : dict
            包含：
            - "conj_lb"
            - "rec_lb"
            - "diff_lb"
            - "x"
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        set_times_paper_style()

        # ------------------ 参数检查 ------------------
        if not isinstance(dimension, int):
            raise TypeError("dimension 必须是 int")
        if dimension < 2:
            raise ValueError("dimension 必须 >= 2")
        if dimension < self.k:
            raise ValueError(f"dimension={dimension} 不能小于 k={self.k}")
        if n_sample is None:
            n_sample = int(self.n_sample)
        if n_sample < 1:
            raise ValueError("n_sample 必须为正整数")

        rng = np.random.default_rng(seed)

        # ------------------ 采样 ------------------
        if use_conf:
            probs = self.sample_conf(current_dimension=dimension, n_sample=n_sample)
        else:
            probs = self.sample(current_dimension=dimension, n_sample=n_sample)

        conj_lb = []
        rec_lb = []

        yi = self.yi
        xj = self.xj

        # ------------------ 逐 sample 计算 lower bound ------------------
        for prob in probs:
            my_PNS = ConjecturePNS(
                n=dimension, k=self.k, p=1, q=1,
                prob=prob, yi=yi, xj=xj
            )
            re_PNS = RecursivePNS(
                n=dimension, m=dimension, k=self.k, p=1, q=1,
                prob=prob, yi=yi, xj=xj
            )

            conj_lb.append(my_PNS.lb)
            rec_lb.append(re_PNS.lb)

        conj_lb = np.asarray(conj_lb, dtype=float)
        rec_lb = np.asarray(rec_lb, dtype=float)
        diff_lb = conj_lb - rec_lb

        # ------------------ 如果只想看差异最大的 top_k ------------------
        if top_k is not None and top_k > 0 and len(diff_lb) > top_k:
            score = np.abs(diff_lb)
            jitter = rng.random(len(score)) * 1e-12
            idx = np.argsort(score + jitter, kind="mergesort")[-top_k:]
            conj_lb = conj_lb[idx]
            rec_lb = rec_lb[idx]
            diff_lb = diff_lb[idx]

        # ------------------ 排序 ------------------
        if sort_by == "diff_abs":
            order = np.argsort(np.abs(diff_lb), kind="mergesort")
        elif sort_by == "diff":
            order = np.argsort(diff_lb, kind="mergesort")
        elif sort_by == "rec_lb":
            order = np.argsort(rec_lb, kind="mergesort")
        elif sort_by == "conj_lb":
            order = np.argsort(conj_lb, kind="mergesort")
        elif sort_by == "none":
            order = np.arange(len(diff_lb))
        else:
            raise ValueError('sort_by 必须是 "diff_abs", "diff", "rec_lb", "conj_lb", "none" 之一')

        conj_lb = conj_lb[order]
        rec_lb = rec_lb[order]
        diff_lb = diff_lb[order]

        x = np.arange(1, len(conj_lb) + 1)

        # ------------------ 打印一些统计量 ------------------
        print(f"[draw_lb_values] dimension={dimension}, n_sample={n_sample}, used={len(x)}")
        print(f"  Thm2-LB min/max   : {conj_lb.min():.6f} / {conj_lb.max():.6f}")
        print(f"  LiPearl-LB min/max: {rec_lb.min():.6f} / {rec_lb.max():.6f}")
        print(f"  diff min/max      : {diff_lb.min():.6f} / {diff_lb.max():.6f}")
        print(f"  mean diff         : {diff_lb.mean():.6f}")
        print(f"  mean abs diff     : {np.mean(np.abs(diff_lb)):.6f}")

        # ------------------ y 轴范围：按值本身 + difference 自动缩放 ------------------
        all_y = np.concatenate([conj_lb, rec_lb])
        y_min = float(all_y.min())
        y_max = float(all_y.max())
        y_range = y_max - y_min

        diff_scale = float(np.max(np.abs(diff_lb))) if len(diff_lb) > 0 else 0.0

        # 关键：padding 不取特别大，而是参考实际 range 和 difference
        pad = max(0.08 * max(y_range, 1e-6), 0.8 * diff_scale, 1e-4)

        # 如果两条线几乎重合，仍然给一点空间
        if y_range < 1e-6:
            center = 0.5 * (y_min + y_max)
            y_min = center - pad
            y_max = center + pad
        else:
            y_min = y_min - pad
            y_max = y_max + pad

        # ------------------ 图1：两种方法的 lower bound 具体值 ------------------
        plt.figure(figsize=(3.25, 2.4))

        plt.plot(x, rec_lb, linestyle='-', color='blue', label='Li and Pearl-LB')
        plt.plot(x, conj_lb, linestyle='-', color='red', label='Thm2-LB')

        ax = plt.gca()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
        ax.margins(x=0.01, y=0.02)
        ax.set_ylim(y_min, y_max)

        plt.xlabel('Samples')
        plt.ylabel('Lower Bounds for PNS')
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()

        if save_prefix is not None:
            plt.savefig(f"{save_prefix}_n{dimension}.pdf", bbox_inches="tight")

        # ------------------ 图2：difference（可选） ------------------
        if show_diff:
            plt.figure(figsize=(3.25, 2.4))

            plt.plot(x, diff_lb, linestyle='-', color='green',
                    label=r'$\Delta LB = LB_{\mathrm{Thm2}} - LB_{\mathrm{Li&Pearl}}$')

            ax2 = plt.gca()
            ax2.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
            ax2.margins(x=0.01, y=0.05)
            plt.axhline(0, linewidth=1, linestyle=':', color='gray')

            # difference 图单独按 difference 自己的范围来缩放
            dmin = float(diff_lb.min())
            dmax = float(diff_lb.max())
            drange = dmax - dmin
            dpad = max(0.1 * max(drange, 1e-6), 1e-5)
            ax2.set_ylim(dmin - dpad, dmax + dpad)

            plt.xlabel('Samples')
            plt.ylabel('Lower Bound Difference')
            plt.legend(loc="best", frameon=True)
            plt.tight_layout()

            if save_prefix is not None:
                plt.savefig(f"{save_prefix}_n{dimension}_diff.pdf", bbox_inches="tight")

        plt.show()

        return {
            "x": x,
            "conj_lb": conj_lb,
            "rec_lb": rec_lb,
            "diff_lb": diff_lb,
        }

    def draw_k_to_n(self):
        # x 轴：维度 2..n_dimension
        set_times_paper_style()

        dims = list(range(self.k, self.n_dimension + 1))

        # 防御：截断到相同长度（避免偶发长度不一致）
        L = min(
            len(dims),
            len(self.lb), len(self.ub), len(self.lbx), len(self.ubx),
            len(self.lby), len(self.uby), len(self.lbxy), len(self.ubxy),
            len(self.rlb), len(self.rub), len(self.rlbx), len(self.rubx),
            len(self.rlby), len(self.ruby), len(self.rlbxy), len(self.rubxy)
        )
        dims = dims[:L]

        # 原猜想（Conjecture）曲线
        y_lb = self.lb[:L]
        y_ub = self.ub[:L]
        y_lbx = self.lbx[:L]
        y_ubx = self.ubx[:L]
        y_lby = self.lby[:L]
        y_uby = self.uby[:L]
        y_lbxy = self.lbxy[:L]
        y_ubxy = self.ubxy[:L]

        # 递归（Recursive）曲线
        ry_lb = self.rlb[:L]
        ry_ub = self.rub[:L]
        ry_lbx = self.rlbx[:L]
        ry_ubx = self.rubx[:L]
        ry_lby = self.rlby[:L]
        ry_uby = self.ruby[:L]
        ry_lbxy = self.rlbxy[:L]
        ry_ubxy = self.rubxy[:L]

        # -----------------------
        # 原四张对比图（黄=Conjecture，黑=Recursive）
        # -----------------------

        # 图1：总体 LB / UB
        plt.figure(figsize=(3.25, 2.4))
        plt.plot(dims, ry_ub, marker='s', linestyle='-', color='black', label='Li and Pearl-UB')
        plt.plot(dims, y_ub, marker='s', linestyle='-', color='green', label='Thm2-UB')
        plt.plot(dims, ry_lb, marker='o', linestyle='-', color='red', label='Li and Pearl-LB')
        plt.plot(dims, y_lb, marker='o', linestyle='-', color='blue', label='Thm2-LB')
        k = self.k
        plt.xticks(np.arange(2, self.n_dimension + 1, 1))
        # plt.title(f'PNS({k}) Bounds')  # 不再手写 fontsize，走全局 rcParams
        plt.xlabel('Dimension')
        plt.ylabel('Bounds for PNS')
        # plt.legend()
        plt.legend(loc="best", frameon=True)
        # plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"bounds_k_to_n.pdf", bbox_inches="tight")  # 不糊（矢量）

        # # 图2：固定 X 的 LBx / UBx
        # plt.figure(figsize=(10, 6))
        # plt.plot(dims, y_lbx, marker='o', linestyle='-', color='gold', label='Conj-LBx')
        # plt.plot(dims, y_ubx, marker='s', linestyle='-', color='gold', label='Conj-UBx')
        # plt.plot(dims, ry_lbx, marker='o', linestyle='--', color='black', label='Rec-LBx')
        # plt.plot(dims, ry_ubx, marker='s', linestyle='--', color='black', label='Rec-UBx')
        # plt.xticks(np.arange(2, self.n_dimension+1, 1))
        # plt.title('Bounds for Fixed X_p vs Dimension', fontsize=TITLE_FT)
        # plt.xlabel('Dimension n', fontsize=LABEL_FT)
        # plt.ylabel('Value', fontsize=LABEL_FT)
        # plt.legend(fontsize=LEGEND_FT)
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()
        #
        # # 图3：固定 Y 的 LBy / UBy
        # plt.figure(figsize=(10, 6))
        # plt.plot(dims, y_lby, marker='o', linestyle='-', color='gold', label='Conj-LBy')
        # plt.plot(dims, y_uby, marker='s', linestyle='-', color='gold', label='Conj-UBy')
        # plt.plot(dims, ry_lby, marker='o', linestyle='--', color='black', label='Rec-LBy')
        # plt.plot(dims, ry_uby, marker='s', linestyle='--', color='black', label='Rec-UBy')
        # plt.xticks(np.arange(2, self.n_dimension+1, 1))
        # plt.title('Bounds for Fixed Y_q vs Dimension', fontsize=TITLE_FT)
        # plt.xlabel('Dimension n', fontsize=LABEL_FT)
        # plt.ylabel('Value', fontsize=LABEL_FT)
        # plt.legend(fontsize=LEGEND_FT)
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()
        #
        # # 图4：固定 (X_p, Y_q) 的 LBxy / UBxy
        # plt.figure(figsize=(10, 6))
        # plt.plot(dims, y_lbxy, marker='o', linestyle='-', color='gold', label='Conj-LBxy')
        # plt.plot(dims, y_ubxy, marker='s', linestyle='-', color='gold', label='Conj-UBxy')
        # plt.plot(dims, ry_lbxy, marker='o', linestyle='--', color='black', label='Rec-LBxy')
        # plt.plot(dims, ry_ubxy, marker='s', linestyle='--', color='black', label='Rec-UBxy')
        # plt.xticks(np.arange(2, self.n_dimension+1, 1))
        # plt.title('Bounds for (X_p, Y_q) vs Dimension', fontsize=TITLE_FT)
        # plt.xlabel('Dimension n', fontsize=LABEL_FT)
        # plt.ylabel('Value', fontsize=LABEL_FT)
        # plt.legend(fontsize=LEGEND_FT)
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()

        # -----------------------
        # 新增四张“差值图”
        # 规则：LB/LBx/LBy/LBxy 用 (Conj − Rec)；UB/UBx/UBy/UBxy 用 (Rec − Conj)
        # -----------------------

        # 预先计算差值（按你的规则）
        d_lb = np.array(y_lb) - np.array(ry_lb)  # conj - rec
        d_ub = np.array(ry_ub) - np.array(y_ub)  # rec - conj
        # d_lbx = np.array(y_lbx) - np.array(ry_lbx)  # conj - rec
        # d_ubx = np.array(ry_ubx) - np.array(y_ubx)  # rec - conj
        # d_lby = np.array(y_lby) - np.array(ry_lby)  # conj - rec
        # d_uby = np.array(ry_uby) - np.array(y_uby)  # rec - conj
        # d_lbxy = np.array(y_lbxy) - np.array(ry_lbxy)  # conj - rec
        # d_ubxy = np.array(ry_ubxy) - np.array(y_ubxy)  # rec - conj

        # 差值图 1：LB / UB 的差
        plt.figure(figsize=(3.25, 2.4))
        plt.plot(dims, d_ub, marker='s', linestyle='-', color='black', label=r'$\Delta UB = \mathrm{Li & Pearl} - \mathrm{Thm2}$')
        plt.plot(dims, d_lb, marker='o', linestyle='-', color='green', label=r'$\Delta LB = \mathrm{Thm2} - \mathrm{Li & Pearl}$')
        plt.xticks(np.arange(2, self.n_dimension + 1, 1))
        # plt.axhline(0, linewidth=1, linestyle=':', color='gray')
        # plt.title(f'PNS({k}) Bounds Diff')
        plt.xlabel('Dimension')
        plt.ylabel('Bounds Differences for PNS')
        # plt.legend()
        plt.legend(loc="best", frameon=True)
        # plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"bounds_k_to_n_diff.pdf", bbox_inches="tight")  # 不糊（矢量）

        # # 差值图 2：LBx / UBx 的差值
        # plt.figure(figsize=(10, 6))
        # plt.plot(dims, d_lbx, marker='o', linestyle='-', label='ΔLBx = Conj − Rec')
        # plt.plot(dims, d_ubx, marker='s', linestyle='--', label='ΔUBx = Rec − Conj')
        # plt.xticks(np.arange(2, self.n_dimension+1, 1))
        # plt.axhline(0, linewidth=1, linestyle=':', color='gray')
        # plt.title('DIFF: Fixed X_p (LBx: Conj−Rec, UBx: Rec−Conj)', fontsize=TITLE_FT)
        # plt.xlabel('Dimension n', fontsize=LABEL_FT)
        # plt.ylabel('Difference', fontsize=LABEL_FT)
        # plt.legend(fontsize=LEGEND_FT)
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()
        #
        # # 差值图 3：LBy / UBy 的差值
        # plt.figure(figsize=(10, 6))
        # plt.plot(dims, d_lby, marker='o', linestyle='-', label='ΔLBy = Conj − Rec')
        # plt.plot(dims, d_uby, marker='s', linestyle='--', label='ΔUBy = Rec − Conj')
        # plt.xticks(np.arange(2, self.n_dimension+1, 1))
        # plt.axhline(0, linewidth=1, linestyle=':', color='gray')
        # plt.title('DIFF: Fixed Y_q (LBy: Conj−Rec, UBy: Rec−Conj)', fontsize=TITLE_FT)
        # plt.xlabel('Dimension n', fontsize=LABEL_FT)
        # plt.ylabel('Difference', fontsize=LABEL_FT)
        # plt.legend(fontsize=LEGEND_FT)
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()
        #
        # # 差值图 4：LBxy / UBxy 的差值
        # plt.figure(figsize=(10, 6))
        # plt.plot(dims, d_lbxy, marker='o', linestyle='-', label='ΔLBxy = Conj − Rec')
        # plt.plot(dims, d_ubxy, marker='s', linestyle='--', label='ΔUBxy = Rec − Conj')
        # plt.xticks(np.arange(2, self.n_dimension+1, 1))
        # plt.axhline(0, linewidth=1, linestyle=':', color='gray')
        # plt.title('DIFF: (X_p, Y_q) (LBxy: Conj−Rec, UBxy: Rec−Conj)', fontsize=TITLE_FT)
        # plt.xlabel('Dimension n', fontsize=LABEL_FT)
        # plt.ylabel('Difference', fontsize=LABEL_FT)
        # plt.legend(fontsize=LEGEND_FT)
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()

        plt.show()

    def test(self):
    # sample:
    # 1: 0
    # 2: 213/300 + 184/300 + 189/300 − 2
    # 3: -0.047
    # 4: 0
        prob_myPNS = {
            # 观测联合分布 P(xi, yj)
            "P(x1,y1)": 131 / 900, "P(x1,y2)": 68 / 900, "P(x1,y3)": 1 / 900,
            "P(x2,y1)": 45 / 900, "P(x2,y2)": 22 / 900, "P(x2,y3)": 51 / 900,
            "P(x3,y1)": 38 / 900, "P(x3,y2)": 483 / 900, "P(x3,y3)": 61 / 900,

            # 实验/反事实分布 P(yj_xi) = P(Y=yj | do(X=xi))
            "P(y1_x1)": 46 / 300, "P(y2_x1)": 23 / 300, "P(y3_x1)": 231 / 300,
            "P(y1_x2)": 270 / 300, "P(y2_x2)": 8 / 300, "P(y3_x2)": 22 / 300,
            "P(y1_x3)": 40 / 300, "P(y2_x3)": 223 / 300, "P(y3_x3)": 37 / 300,
        }

        my_PNS = ConjecturePNS(n=3, k=3, p=2, q=1, prob=prob_myPNS, yi=[3,1,2], xj=[1,2,3])
        print(my_PNS.lb)
        print(my_PNS.ub)

        prob1_recPNS = {
            # 观测联合分布 P(xi, yj)
            "P(x1,y1)": 238 / 900, "P(x1,y2)": 20 / 900, "P(x1,y3)": 7 / 900,
            "P(x2,y1)": 10 / 900, "P(x2,y2)": 77 / 900, "P(x2,y3)": 259 / 900,
            "P(x3,y1)": 147 / 900, "P(x3,y2)": 72 / 900, "P(x3,y3)": 70 / 900,

            # 实验/反事实分布 P(yj_xi) = P(Y=yj | do(X=xi))
            "P(y1_x1)": 80 / 300, "P(y2_x1)": 7 / 300, "P(y3_x1)": 213 / 300,
            "P(y1_x2)": 184 / 300, "P(y2_x2)": 29 / 300, "P(y3_x2)": 87 / 300,
            "P(y1_x3)": 87 / 300, "P(y2_x3)": 189 / 300, "P(y3_x3)": 24 / 300,
        }

        prob2_recPNS = {
            # 观测联合分布 P(xi, yj)
            "P(x1,y1)": 92 / 1200, "P(x1,y2)": 58 / 1200,
            "P(x2,y1)": 55 / 1200, "P(x2,y2)": 118 / 1200,
            "P(x3,y1)": 24 / 1200, "P(x3,y2)": 231 / 1200,
            "P(x4,y1)": 599 / 1200, "P(x4,y2)": 23 / 1200,

            # 实验/反事实分布 P(yj_xi) = P(Y=yj | do(X=xi))
            "P(y1_x1)": 53 / 300, "P(y2_x1)": 247 / 300,
            "P(y1_x2)": 269 / 300, "P(y2_x2)": 31 / 300,
            "P(y1_x3)": 234 / 300, "P(y2_x3)": 66 / 300,
            "P(y1_x4)": 151 / 300, "P(y2_x4)": 149 / 300,
        }

        re_PNS1 = RecursivePNS(n=3, m=3, k=3, p=2, q=1, prob=prob1_recPNS, yi=[3,1,2], xj=[1,2,3])
        re_PNS2 = RecursivePNS(n=2, m=4, k=1, p=2, q=2, prob=prob2_recPNS, yi=[1], xj=[4])
        # print(re_PNS1.lb)
        # print(re_PNS1.ub)
        # print(re_PNS2.lbxy/(118 / 1200))
        # print(re_PNS2.ubxy/(118 / 1200))
