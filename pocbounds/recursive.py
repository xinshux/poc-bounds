"""Recursive (Li and Pearl) baseline bounds, used for comparison.
"""

from itertools import combinations


class RecursivePNS:
    lb = []
    lbx = []
    lby = []
    lbxy = []
    ub = []
    ubx = []
    uby = []
    ubxy = []
    def __init__(self,n,m,k,p,q,prob={},yi=[],xj=[]):
        self.prob = prob
        # print(prob)
        self.n = n
        self.m = m
        self.k = k
        self.p = p
        self.q = q
        self.yi = yi
        self.xj = xj

        self.lb = self.PNS_rlb(n, m, k, p, q, z=1, prob=prob, yi=yi,xj=xj)
        self.ub = self.PNS_rub(n, m, k, p, q, z=1, prob=prob, yi=yi,xj=xj)
        self.lbx = self.PNS_rlb(n, m, k, p, q, z=2, prob=prob, yi=yi,xj=xj)
        self.ubx = self.PNS_rub(n, m, k, p, q, z=2, prob=prob, yi=yi,xj=xj)
        self.lby = self.PNS_rlb(n, m, k, p, q, z=3, prob=prob, yi=yi,xj=xj)
        self.uby = self.PNS_rub(n, m, k, p, q, z=3, prob=prob, yi=yi,xj=xj)
        self.lbxy = self.PNS_rlb(n, m, k, p, q, z=4, prob=prob, yi=yi,xj=xj)
        self.ubxy = self.PNS_rub(n, m, k, p, q, z=4, prob=prob, yi=yi,xj=xj)

        # # give all values we need
        # self.PNS_rlb(n, m, k, p, q, z, prob, yi,xj)
        # self.PNS_rub(n, m, k, p, q, z, prob, yi,xj)

    def lbx_base(self,n,m,k,p,q,z,prob,y_index,x_index):
        if x_index != p:
            lbx_base0 = 0
            lbx_base1 = prob[f"P(y{y_index}_x{x_index})"]-prob[f"P(x{x_index},y{y_index})"] - 1
            for i in range(n):
                lbx_base1 += prob[f"P(x{p},y{i+1})"]
            for i in range(n):
                lbx_base1 += prob[f"P(x{x_index},y{i+1})"]
            return max(lbx_base0,lbx_base1)
        else:
            return prob[f"P(x{x_index},y{y_index})"]
    def ubx_base(self,n,m,k,p,q,z,prob,y_index,x_index):
        if x_index != p:
            ubx_base0 = prob[f"P(y{y_index}_x{x_index})"]-prob[f"P(x{x_index},y{y_index})"]
            ubx_base1 = 0
            for i in range(n):
                ubx_base1 += prob[f"P(x{p},y{i+1})"]
            return min(ubx_base0, ubx_base1)
        else:
            return prob[f"P(x{x_index},y{y_index})"]
    def lby_base(self,n,m,k,p,q,z,prob,y_index,x_index):
        if y_index == q:
            lbysame_base0 = prob[f"P(x{x_index},y{y_index})"]
            lbysame_base1 = prob[f"P(y{y_index}_x{x_index})"] - 1
            for i in range(m):
                lbysame_base1 += prob[f"P(x{i+1},y{y_index})"]
            return max(lbysame_base0,lbysame_base1)
        elif y_index != q:
            lbydiff_base0 = 0
            lbydiff_base1 = prob[f"P(y{y_index}_x{x_index})"] - 1
            for i in range(m):
                lbydiff_base1 += prob[f"P(x{i+1},y{q})"]
            lbydiff_base2 = [0] * m
            for p_temp in range(m):
                if p_temp+1 != x_index:
                    val = prob[f"P(y{y_index}_x{x_index})"] - prob[f"P(x{x_index},y{y_index})"] - 1 + prob[f"P(x{p_temp + 1},y{q})"]
                    for i in range(n):
                        val += prob[f"P(x{x_index},y{i+1})"]
                    if val > 0:
                        lbydiff_base2[p_temp] = val
            return max(lbydiff_base0, lbydiff_base1, sum(lbydiff_base2))
    def uby_base(self,n,m,k,p,q,z,prob,y_index,x_index):
        if y_index == q:
            ubysame_base0 = prob[f"P(y{y_index}_x{x_index})"]
            ubysame_base1 = 0
            for i in range(m):
                ubysame_base1 += prob[f"P(x{i+1},y{y_index})"]
            return min(ubysame_base0, ubysame_base1)
        elif y_index != q:
            ubydiff_base0 = prob[f"P(y{y_index}_x{x_index})"] - prob[f"P(x{x_index},y{y_index})"]
            ubydiff_base1 = - prob[f"P(x{x_index},y{q})"]
            for i in range(m):
                ubydiff_base1 += prob[f"P(x{i+1},y{q})"]
            return min(ubydiff_base0, ubydiff_base1)
    def lbxy_base(self,n,m,k,p,q,z,prob,y_index,x_index):
        if x_index != p:
            lbxy_base0 = 0
            lbxy_base1 = prob[f"P(y{y_index}_x{x_index})"] - prob[f"P(x{x_index},y{y_index})"] - 1 + prob[f"P(x{p},y{q})"]
            for i in range(n):
                lbxy_base1 += prob[f"P(x{x_index},y{i+1})"]
            return max(lbxy_base0, lbxy_base1)
        else:
            if y_index == q:
                return prob[f"P(x{x_index},y{y_index})"]
            else:
                return 0
    def ubxy_base(self,n,m,k,p,q,z,prob,y_index,x_index):
        if x_index != p:
            ubxy_base0 = prob[f"P(y{y_index}_x{x_index})"]-prob[f"P(x{x_index},y{y_index})"]
            ubxy_base1 = prob[f"P(x{p},y{q})"]
            return min(ubxy_base0, ubxy_base1)
        else:
            if y_index == q:
                return prob[f"P(x{x_index},y{y_index})"]
            else:
                return 0

    def PNS_rlb(self,n,m,k,p,q,z,prob,yi,xj):
        # apply base case
        if k == 1:
            if z == 1: # no_add
                return prob[f"P(y{yi[0]}_x{xj[0]})"]
            elif z == 2: # +x
                return self.lbx_base(n, m, k, p=p, q=q, z=z, prob=prob, y_index=yi[0], x_index=xj[0])
            elif z == 3: # +y
                return self.lby_base(n, m=m, k=k, p=p, q=q, z=z, prob=prob, y_index=yi[0], x_index=xj[0])
            elif z == 4: # +x+y
                return self.lbxy_base(n, m, k, p=p, q=q, z=z, prob=prob, y_index=yi[0], x_index=xj[0])
        # z=1 means lb
        elif z == 1:
            rlb0 = 0
            rlb1 = -k+1
            for t in range(k):
                rlb1 += prob[f"P(y{yi[t]}_x{xj[t]})"]
            rlb2 = [0]*k
            for t in range(k):
                new_yi = yi.copy()
                new_xj = xj.copy()
                del new_yi[t]
                del new_xj[t]
                rlb2[t] = self.PNS_rlb(n,m,k-1,p,q,1,prob,new_yi,new_xj) + prob[f"P(y{yi[t]}_x{xj[t]})"] - 1
            rlb3 = 0
            for r in range(m):
                new_yi = yi.copy()
                new_xj = xj.copy()
                if r+1 in new_xj:
                    idx = new_xj.index(r+1)
                    p = new_xj[idx]
                    q = new_yi[idx]
                    del new_yi[idx]
                    del new_xj[idx]
                    rlb3 += self.PNS_rlb(n,m,k-1,p,q,4,prob,new_yi,new_xj)
                else:
                    p = r+1
                    rlb3 += self.PNS_rlb(n,m,k,p,q,2,prob,new_yi,new_xj)
            return max(rlb0, rlb1, max(rlb2), rlb3)
        # z=2 means lbx
        elif z == 2:
            if p not in xj:
                rlb0 = 0
                rlb1 = -k
                for i in range(n):
                    rlb1 += prob[f"P(x{p},y{i + 1})"]
                for t in range(k):
                    rlb1 += prob[f"P(y{yi[t]}_x{xj[t]})"]
                rlb2 = [0] * k
                for t in range(k):
                    new_yi = yi.copy()
                    new_xj = xj.copy()
                    y_index = new_yi[t]
                    x_index = new_xj[t]
                    del new_yi[t]
                    del new_xj[t]
                    rlb2[t] = self.PNS_rlb(n, m, k-1, p, q,1, prob, new_yi, new_xj) + self.lbx_base(n,m,k,p,q,z,prob,y_index,x_index) - 1
                return max(rlb0, rlb1, max(rlb2))
            else:
                new_yi = yi.copy()
                new_xj = xj.copy()
                idx = new_xj.index(p)
                p = new_xj[idx]
                q = new_yi[idx]
                del new_yi[idx]
                del new_xj[idx]
                return self.PNS_rlb(n, m, k-1, p, q,4, prob, new_yi, new_xj)
        # z=3 means lby
        elif z == 3:
            rlb0 = 0
            rlb1 = -k
            for i in range(m):
                rlb1 += prob[f"P(x{i+1},y{q})"]
            for t in range(k):
                rlb1 += prob[f"P(y{yi[t]}_x{xj[t]})"]
            rlb2 = [0] * k
            for t in range(k):
                new_yi = yi.copy()
                new_xj = xj.copy()
                y_index = new_yi[t]
                x_index = new_xj[t]
                del new_yi[t]
                del new_xj[t]
                rlb2[t] = self.PNS_rlb(n, m, k - 1, p, q, 1, prob, new_yi, new_xj) + self.lby_base(n,m,k,p,q,z,prob,y_index,x_index) - 1
            rlb3 = 0
            for r in range(m):
                new_yi = yi.copy()
                new_xj = xj.copy()
                if r+1 in new_xj:
                    idx = new_xj.index(r+1)
                    if new_yi[idx] == q:
                        p = new_xj[idx]
                        del new_yi[idx]
                        del new_xj[idx]
                        rlb3 += self.PNS_rlb(n,m,k-1,p,q,4,prob,new_yi,new_xj)
                else:
                    p = r+1
                    rlb3 += self.PNS_rlb(n,m,k,p,q,4,prob,new_yi,new_xj)
            return max(rlb0, rlb1, max(rlb2), rlb3)
        # z=4 means lbxy
        elif z == 4:
            if p not in xj:
                rlb0 = 0
                rlb1 = prob[f"P(x{p},y{q})"]-k
                for t in range(k):
                    rlb1 += prob[f"P(y{yi[t]}_x{xj[t]})"]
                rlb2 = [0] * k
                for t in range(k):
                    new_yi = yi.copy()
                    new_xj = xj.copy()
                    y_index = new_yi[t]
                    x_index = new_xj[t]
                    del new_yi[t]
                    del new_xj[t]
                    rlb2[t] = self.PNS_rlb(n, m, k - 1, p, q, 1, prob, new_yi, new_xj) + self.lbxy_base(n, m, k, p, q, z, prob, y_index, x_index) - 1
                return max(rlb0, rlb1, max(rlb2))
            else:
                new_yi = yi.copy()
                new_xj = xj.copy()
                idx = new_xj.index(p)
                if q == new_yi[idx]:
                    p = new_xj[idx]
                    q = new_yi[idx]
                    del new_yi[idx]
                    del new_xj[idx]
                    return self.PNS_rlb(n, m, k - 1, p, q, 4, prob, new_yi, new_xj)
                else:
                    return 0

    def PNS_rub(self, n, m, k, p, q, z, prob, yi, xj):
        # apply base case
        if k == 1:
            if z == 1:  # no_add
                return prob[f"P(y{yi[0]}_x{xj[0]})"]
            elif z == 2:  # +x
                return self.ubx_base(n, m, k, p=p, q=q, z=z, prob=prob, y_index=yi[0], x_index=xj[0])
            elif z == 3:  # +y
                return self.uby_base(n, m, k, p, q=q, z=z, prob=prob, y_index=yi[0], x_index=xj[0])
            elif z == 4:  # +x+y
                return self.ubxy_base(n, m, k, p=p, q=q, z=z, prob=prob, y_index=yi[0], x_index=xj[0])
        # z=1 means ub
        elif z == 1:
            rub0 = [100000]*k
            for t in range(k):
                rub0[t] = prob[f"P(y{yi[t]}_x{xj[t]})"]
            rub1 = [100000]*k
            for t in range(k):
                new_yi = yi.copy()
                new_xj = xj.copy()
                del new_yi[t]
                del new_xj[t]
                rub1[t] = self.PNS_rub(n, m, k-1, p, q, 1, prob, new_yi, new_xj)
            rub2 = 0
            for r in range(m):
                new_yi = yi.copy()
                new_xj = xj.copy()
                if r+1 in new_xj:
                    idx = new_xj.index(r+1)
                    p = new_xj[idx]
                    q = new_yi[idx]
                    del new_yi[idx]
                    del new_xj[idx]
                    rub2 += self.PNS_rub(n, m, k - 1, p, q, 4, prob, new_yi, new_xj)
                else:
                    p = r+1
                    rub2 += self.PNS_rub(n, m, k, p, q, 2, prob, new_yi, new_xj)
            return min(min(rub0), min(rub1), rub2)
        # z=2 means ubx
        elif z == 2:
            if p not in xj:
                rub0 = [100000] * k
                for t in range(k):
                    rub0[t] = prob[f"P(y{yi[t]}_x{xj[t]})"]
                rub1 = 0
                for i in range(n):
                    rub1 += prob[f"P(x{p},y{i + 1})"]
                rub2 = [100000] * k
                for t in range(k):
                    new_yi = yi.copy()
                    new_xj = xj.copy()
                    del new_yi[t]
                    del new_xj[t]
                    rub2[t] = self.PNS_rub(n, m, k - 1, p, q, 1, prob, new_yi, new_xj)
                rub3 = [100000] * k
                for t in range(k):
                    y_index = yi[t]
                    x_index = xj[t]
                    rub3[t] = self.ubx_base(n, m, k, p, q, z, prob, y_index, x_index)
                return min(min(rub0), rub1, min(rub2), min(rub3))
            else:
                new_yi = yi.copy()
                new_xj = xj.copy()
                idx = new_xj.index(p)
                p = new_xj[idx]
                q = new_yi[idx]
                del new_yi[idx]
                del new_xj[idx]
                return self.PNS_rub(n, m, k-1, p, q,4, prob, new_yi, new_xj)
        # z=3 means uby
        elif z == 3:
            rub0 = [100000] * k
            for t in range(k):
                rub0[t] = prob[f"P(y{yi[t]}_x{xj[t]})"]
            rub1 = 0
            for i in range(m):
                rub1 += prob[f"P(x{i+1},y{q})"]
            rub2 = [10000]*k
            for t in range(k):
                new_yi = yi.copy()
                new_xj = xj.copy()
                del new_yi[t]
                del new_xj[t]
                rub2[t] = self.PNS_rub(n, m, k - 1, p, q, 1, prob, new_yi, new_xj)
            rub3 = [10000]*k
            for t in range(k):
                y_index = yi[t]
                x_index = xj[t]
                rub3[t] = self.uby_base(n,m,k,p,q,z,prob,y_index,x_index)
            rub4 = 0
            for r in range(m):
                new_yi = yi.copy()
                new_xj = xj.copy()
                if r+1 in new_xj:
                    idx = new_xj.index(r+1)
                    if new_yi[idx] == q:
                        p = new_xj[idx]
                        del new_yi[idx]
                        del new_xj[idx]
                        rub4 += self.PNS_rub(n, m, k - 1, p, q, 4, prob, new_yi, new_xj)
                else:
                    p = r+1
                    rub4 += self.PNS_rub(n, m, k, p, q, 4, prob, new_yi, new_xj)
            return min(min(rub0), rub1, min(rub2), min(rub3), rub4)
        # z=4 means ubxy
        elif z == 4:
            if p not in xj:
                rub0 = [100000] * k
                for t in range(k):
                    rub0[t] = prob[f"P(y{yi[t]}_x{xj[t]})"]
                rub1 = prob[f"P(x{p},y{q})"]
                rub2 = [10000] * k
                for t in range(k):
                    new_yi = yi.copy()
                    new_xj = xj.copy()
                    del new_yi[t]
                    del new_xj[t]
                    rub2[t] = self.PNS_rub(n, m, k - 1, p, q, 1, prob, new_yi, new_xj)
                rub3 = [10000] * k
                for t in range(k):
                    y_index = yi[t]
                    x_index = xj[t]
                    rub3[t] = self.ubxy_base(n, m, k, p, q, z, prob, y_index, x_index)
                return min(min(rub0), rub1, min(rub2), min(rub3))
            else:
                # print(f"p:{p}, q:{q},old_yi:{yi}, old_xj:{xj}")
                new_yi = yi.copy()
                new_xj = xj.copy()
                idx = new_xj.index(p)
                if q == new_yi[idx]:
                    p = new_xj[idx]
                    q = new_yi[idx]
                    del new_yi[idx]
                    del new_xj[idx]
                    # print(f"new_yi:{new_yi}, new_xj:{new_xj}, k-1:{k-1}")
                    return self.PNS_rub(n, m, k - 1, p, q, 4, prob, new_yi, new_xj)
                else:
                    return 0
