"""Closed-form (conjectured) bounds on multivalued probabilities of causation.

`ConjecturePNS` implements the bounds of Shu, Wang and Li (IJCAI-ECAI 2026)
for PNS(k), PSub(k, p), PRep(k, q) and PN(k, p, q).  After construction the
eight attributes ``lb``/``ub``, ``lbx``/``ubx``, ``lby``/``uby`` and
``lbxy``/``ubxy`` hold the computed bounds as floats.
"""

from itertools import combinations


class ConjecturePNS:
    # calculate prob for n dimensions
    # def __init__(self,n,prob={}):
    #     self.prob = prob
    #     lb1 = 0
    #     lb2 = -n + 1
    #     for j in range(1,n+1):
    #         lb2 += prob[f"P(y{j}_x{j})"]
    #     lb3 = [-n + 1] * n
    #     for j in range(1,n+1):
    #         lb3 += prob[f"P(y{j}_x{j})"]
    #         for l in range(1,n+1):
    #             lb3 += prob[f"P(x{j},y{l})"]
    #         lb3 -= prob[f"P(x{j},y{j})"]
    #     print()

    def __init__(self,n,k,p,q,prob={},yi=[],xj=[]):
        self.prob = prob
        self.n = n
        self.k = k
        self.p = p
        self.q = q
        self.yi = yi
        self.xj = xj
        self.lb = self._compute_lb(prob, n, k, p, q, yi, xj)
        self.ub = self._compute_ub(prob, n, k, p, q, yi, xj)
        self.lbx = self._compute_lbx(prob, n, k, p, q, yi, xj)
        self.ubx = self._compute_ubx(prob, n, k, p, q, yi, xj)
        self.lby = self._compute_lby(prob, n, k, p, q, yi, xj)
        self.uby = self._compute_uby(prob, n, k, p, q, yi, xj)
        self.lbxy = self._compute_lbxy(prob, n, k, p, q, yi, xj)
        self.ubxy = self._compute_ubxy(prob, n, k, p, q, yi, xj)

    def _compute_lb(self, prob, n, k, p, q, yi, xj):
        # prob = self.prob
        # n = self.n
        # k = self.k
        # yi = self.yi
        # xj = self.xj
        lb1 = 0
        lb2 = -k + 1
        for j in range(k):
            lb2 += prob[f"P(y{yi[j]}_x{xj[j]})"]
        lb3 = [-k + 1] * k
        for i in range(k):
            lb3[i] += prob[f"P(x{xj[i]},y{yi[i]})"]
            for j in range(k):
                if j != i:
                    lb3[i] += prob[f"P(y{yi[j]}_x{xj[j]})"]
                    for l in range(n):
                        lb3[i] += prob[f"P(x{xj[j]},y{l+1})"]
                    lb3[i] -= prob[f"P(x{xj[j]},y{yi[j]})"]
        lb = max(lb1, lb2, max(lb3))
        return lb

    def _compute_ub(self, prob, n, k, p, q, yi, xj):
        # prob = self.prob
        # n = self.n
        # k = self.k
        # yi = self.yi
        # xj = self.xj
        ub1 = 0
        for j in range(k):
            ub1 += prob[f"P(x{xj[j]},y{yi[j]})"]
        temp_ = [i + 1 for i in range(n) if (i + 1) not in xj]
        for temp in temp_:
            for l in range(n):
                ub1 += prob[f"P(x{temp},y{l + 1})"]
        ub2 = [1000000] * k
        for j in range(k):
            ub2[j] = prob[f"P(y{yi[j]}_x{xj[j]})"]
        # ub3 = [[]] * (k-1)
        ub3 = [[] for _ in range(k - 1)]
        # # 1) 逐个遍历所有不重复组合（大小为 l）
        # for combo in combinations(range(k), l):
        #     print(combo)  # e.g. (0,1), (0,2), ...
        # # 2) 一次性拿到所有组合（谨慎：k 很大时会占内存）
        # all_combos = list(combinations(range(k), l))
        for m in range(k - 1):
            temp = []
            for combo in combinations(range(k), m + 2):
                num = 0
                for j in combo:
                    num += (prob[f"P(y{yi[j]}_x{xj[j]})"] - prob[f"P(x{xj[j]},y{yi[j]})"])
                temp.append(num)
            ub3[m] = temp
            for item in ub3[m]:
                item /= (m + 1)
        ub = min(ub1, min(ub2), min([min(ub3[i]) for i in range(k-1)]))
        return ub

    def _compute_lbx(self, prob, n, k, p, q, yi, xj):
        # prob = self.prob
        # n = self.n
        # k = self.k
        # p = self.p
        # yi = self.yi
        # xj = self.xj
        if p not in xj:
            lbx1 = 0
            lbx2 = -k
            for l in range(n):
                lbx2 += prob[f"P(x{p},y{l + 1})"]
            for j in range(k):
                lbx2 += prob[f"P(y{yi[j]}_x{xj[j]})"]
                for l in range(n):
                    lbx2 += prob[f"P(x{xj[j]},y{l + 1})"]
                lbx2 -= prob[f"P(x{xj[j]},y{yi[j]})"]
            lbx = max(lbx1, lbx2)
            return lbx
        else:
            new_yi = yi.copy()
            new_xj = xj.copy()
            idx = new_xj.index(p)
            p = new_xj[idx]
            q = new_yi[idx]
            del new_yi[idx]
            del new_xj[idx]
            return self._compute_lbxy(prob, n, k-1, p, q, new_yi, new_xj)

    def _compute_ubx(self, prob, n, k, p, q, yi, xj):
        # prob = self.prob
        # n = self.n
        # k = self.k
        # p = self.p
        # yi = self.yi
        # xj = self.xj
        if p not in xj:
            ubx1 = 0
            for l in range(n):
                ubx1 += prob[f"P(x{p},y{l + 1})"]
            ubx2 = [1000000]*k
            for j in range(k):
                ubx2[j] = prob[f"P(y{yi[j]}_x{xj[j]})"] - prob[f"P(x{xj[j]},y{yi[j]})"]
            ubx = min(ubx1, min(ubx2))
            return ubx
        else:
            new_yi = yi.copy()
            new_xj = xj.copy()
            idx = new_xj.index(p)
            p = new_xj[idx]
            q = new_yi[idx]
            del new_yi[idx]
            del new_xj[idx]
            return self._compute_ubxy(prob, n, k-1, p, q, new_yi, new_xj)

    def _compute_lby(self, prob, n, k, p, q, yi, xj):
        # prob = self.prob
        # n = self.n
        # k = self.k
        # q = self.q
        # yi = self.yi
        # xj = self.xj
        lby1 = 0
        lby2 = 0
        temp_ = [i+1 for i in range(n) if (i+1) not in xj]
        lby2 = prob[f"P(x{q},y{q})"] - k
        for j in range(k):
            lby2 += prob[f"P(y{yi[j]}_x{xj[j]})"]
            for l in range(n):
                lby2 += prob[f"P(x{xj[j]},y{l + 1})"]
            lby2 -= prob[f"P(x{xj[j]},y{yi[j]})"]
        for temp in temp_:
            if temp != q:
                lby2 += prob[f"P(x{temp},y{q})"]
        lby3 = 0
        if q in xj:
            lby3 = prob[f"P(x{q},y{q})"] - k + 1
            for j in range(k):
                if yi[j] != q:
                    lby3 += prob[f"P(y{yi[j]}_x{xj[j]})"]
                    for l in range(n):
                        lby3 += prob[f"P(x{xj[j]},y{l + 1})"]
                    lby3 -= prob[f"P(x{xj[j]},y{yi[j]})"]
        lby = max(lby1, lby2, lby3)
        return lby

    def _compute_uby(self, prob, n, k, p, q, yi, xj):
        # prob = self.prob
        # n = self.n
        # k = self.k
        # q = self.q
        # yi = self.yi
        # xj = self.xj
        uby1 = 1000000
        if q in xj:
            uby1 = prob[f"P(y{q}_x{q})"]
        uby2 = prob[f"P(x{q},y{q})"]
        temp_ = [i + 1 for i in range(n) if (i + 1) not in xj]
        for temp in temp_:
            if temp != q:
                uby2 += prob[f"P(x{temp},y{q})"]
        uby3 = [1000000]*(k)
        for j in range(k):
            if xj[j] != q:
                uby3[j] = prob[f"P(y{yi[j]}_x{xj[j]})"] - prob[f"P(x{xj[j]},y{yi[j]})"]
        uby = min(uby1, uby2, min(uby3))
        return uby

    def _compute_lbxy(self, prob, n, k, p, q, yi, xj):
        if p not in xj:
            lbxy1 = 0
            lbxy2 = prob[f"P(x{p},y{q})"]-k
            for j in range(k):
                lbxy2 += prob[f"P(y{yi[j]}_x{xj[j]})"]
                for l in range(n):
                    lbxy2 += prob[f"P(x{xj[j]},y{l + 1})"]
                lbxy2 -= prob[f"P(x{xj[j]},y{yi[j]})"]
            lbxy = max(lbxy1, lbxy2)
            return lbxy
        else:
            new_yi = yi.copy()
            new_xj = xj.copy()
            idx = new_xj.index(p)
            if new_yi[idx] == q:
                p = new_xj[idx]
                q = new_yi[idx]
                del new_yi[idx]
                del new_xj[idx]
                return self._compute_lbxy(prob, n, k-1, p, q, new_yi, new_xj)
            else:
                return 0

    def _compute_ubxy(self, prob, n, k, p, q, yi, xj):
        # prob = self.prob
        # n = self.n
        # k = self.k
        # p = self.p
        # q = self.q
        # yi = self.yi
        # xj = self.xj
        if p not in xj:
            ubxy1 = prob[f"P(x{p},y{q})"]
            ubxy2 = [1000000]*k
            for j in range(k):
                ubxy2[j] = prob[f"P(y{yi[j]}_x{xj[j]})"] - prob[f"P(x{xj[j]},y{yi[j]})"]
            ubxy = min(ubxy1, min(ubxy2))
            return ubxy
        else:
            new_yi = yi.copy()
            new_xj = xj.copy()
            idx = new_xj.index(p)
            if new_yi[idx] == q:
                p = new_xj[idx]
                q = new_yi[idx]
                del new_yi[idx]
                del new_xj[idx]
                return self._compute_ubxy(prob, n, k-1, p, q, new_yi, new_xj)
            else:
                return 0
