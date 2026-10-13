"""Random SCM samplers.

Each generator draws a structural causal model of the requested shape and
returns the induced observational and experimental distributions as a dict
with keys ``P(xi,yj)`` and ``P(yj_xi)``.  `generate_prob_with_constraint`
wraps them in rejection sampling so that the returned distribution satisfies
P(x_i, y_j) <= P(y_j | do(x_i)) <= 1 - P(x_i, not y_j).
"""

import numpy as np


def generate_prob_mediator(n_x, n_y=None, n_z=3):
    """
    生成一个概率字典 prob，使得：
      1) p(x_i) >= 0, sum_i p(x_i) = 1
      2) p(z_k | x_i) >= 0, 对每个 i 有 sum_k p(z_k|x_i) = 1
      3) p(y_j | x_i, z_k) >= 0, 对每个 (i,k) 有 sum_j p(y_j|x_i,z_k) = 1

    然后根据：
      P(x_i, y_j) = sum_k p(x_i) p(z_k|x_i) p(y_j|x_i, z_k)
      P(y_j_x_i) = sum_k p(z_k|x_i) p(y_j|x_i, z_k)

    构造出：
      prob["P(xi,yj)"], prob["P(yj_xi)"].
    """

    if n_y is None:
        n_y = n_x

    # 1) p(x)
    p_x = np.random.dirichlet(np.ones(n_x))              # (n_x,)

    # 2) p(z | x)
    p_z_given_x = np.zeros((n_x, n_z))
    for i in range(n_x):
        p_z_given_x[i] = np.random.dirichlet(np.ones(n_z))

    # 3) p(y | x, z)
    p_y_given_xz = np.zeros((n_x, n_z, n_y))
    for i in range(n_x):
        for k in range(n_z):
            p_y_given_xz[i, k] = np.random.dirichlet(np.ones(n_y))

    # 4) 根据公式计算 P(x,y) 和 P(y_x)
    Pxy = np.zeros((n_x, n_y))
    Q = np.zeros((n_x, n_y))          # Q[i,j] = P(Y=y_j | do(X=x_i))

    for i in range(n_x):
        # 先算 Q(i,*):   sum_k p(z_k|x_i) p(y_*|x_i, z_k)
        for k in range(n_z):
            Q[i, :] += p_z_given_x[i, k] * p_y_given_xz[i, k, :]

        # 再算 P(x_i, *): p(x_i) * Q(i,*)
        Pxy[i, :] = p_x[i] * Q[i, :]

    prob = {}
    for i in range(n_x):
        for j in range(n_y):
            prob[f"P(x{i + 1},y{j + 1})"] = float(Pxy[i, j])
            prob[f"P(y{j + 1}_x{i + 1})"] = float(Q[i, j])

    return prob

def generate_prob_confounder(n_x, n_y=None, n_z=3):
    """
    Confounder 结构：Z -> X, Z -> Y, X -> Y
    """

    if n_y is None:
        n_y = n_x

    # 1) p(z)
    p_z = np.random.dirichlet(np.ones(n_z))      # shape: (n_z,)

    # 2) p(x | z)
    p_x_given_z = np.zeros((n_z, n_x))
    for k in range(n_z):
        p_x_given_z[k] = np.random.dirichlet(np.ones(n_x))

    # 3) p(y | x, z)
    p_y_given_xz = np.zeros((n_x, n_z, n_y))
    for i in range(n_x):
        for k in range(n_z):
            p_y_given_xz[i, k] = np.random.dirichlet(np.ones(n_y))

    # 4) 由上面三个生成 P(x,y) 和 P(y_x)
    Pxy = np.zeros((n_x, n_y))
    Q   = np.zeros((n_x, n_y))

    for i in range(n_x):
        for j in range(n_y):
            tmp_xy = 0.0
            tmp_yx = 0.0
            for k in range(n_z):
                tmp_xy += p_z[k] * p_x_given_z[k, i] * p_y_given_xz[i, k, j]
                tmp_yx += p_z[k] * p_y_given_xz[i, k, j]
            Pxy[i, j] = tmp_xy
            Q[i, j]   = tmp_yx

    prob = {}
    for i in range(n_x):
        for j in range(n_y):
            prob[f"P(x{i+1},y{j+1})"] = float(Pxy[i, j])
            prob[f"P(y{j+1}_x{i+1})"] = float(Q[i, j])

    return prob

def generate_prob_complex(n_x, n_y=None, n_z=3):
    """
    Complex 结构，对应 DAG:
        X -> W, X -> Y
        W -> Y, W -> Z
        Y -> Z, Y -> T
        Z -> T
    """

    if n_y is None:
        n_y = n_x

    n_w = n_z
    n_y_mid = n_z

    # 1) P(x)
    p_x = np.random.dirichlet(np.ones(n_x))

    # 2) P(w | x)
    p_w_given_x = np.zeros((n_x, n_w))
    for i in range(n_x):
        p_w_given_x[i] = np.random.dirichlet(np.ones(n_w))

    # 3) P(y_mid | x, w)
    p_y_given_xw = np.zeros((n_x, n_w, n_y_mid))
    for i in range(n_x):
        for w in range(n_w):
            p_y_given_xw[i, w] = np.random.dirichlet(np.ones(n_y_mid))

    # 4) P(z | y_mid, w)
    p_z_given_yw = np.zeros((n_y_mid, n_w, n_z))
    for y_mid in range(n_y_mid):
        for w in range(n_w):
            p_z_given_yw[y_mid, w] = np.random.dirichlet(np.ones(n_z))

    # 5) P(t | y_mid, z)
    p_t_given_yz = np.zeros((n_y_mid, n_z, n_y))
    for y_mid in range(n_y_mid):
        for z in range(n_z):
            p_t_given_yz[y_mid, z] = np.random.dirichlet(np.ones(n_y))

    # 6) Q[i,j] = P(T=t_j | do(X=x_i))
    Q = np.zeros((n_x, n_y))
    for i in range(n_x):
        for w in range(n_w):
            for y_mid in range(n_y_mid):
                for z in range(n_z):
                    Q[i, :] += (
                        p_w_given_x[i, w]
                        * p_y_given_xw[i, w, y_mid]
                        * p_z_given_yw[y_mid, w, z]
                        * p_t_given_yz[y_mid, z, :]
                    )

    # 7) P(x_i, t_j)
    Pxt = np.zeros((n_x, n_y))
    for i in range(n_x):
        Pxt[i, :] = p_x[i] * Q[i, :]

    # 8) 拼成 prob 字典
    prob = {}
    for i in range(n_x):
        for j in range(n_y):
            prob[f"P(x{i + 1},y{j + 1})"] = float(Pxt[i, j])
            prob[f"P(y{j + 1}_x{i + 1})"] = float(Q[i, j])

    return prob

def check_px_y_ineq(prob, n_x, n_y, tol=1e-9):
    """
    检查对所有 i,j 是否满足：
        P(x_i, y_j) <= P(y_j_x_i) <= 1 - P(x_i, not y_j)
    """
    for i in range(n_x):
        for j in range(n_y):
            p_xy = prob[f"P(x{i+1},y{j+1})"]
            p_yx = prob[f"P(y{j+1}_x{i+1})"]

            # P(x_i, not y_j)
            p_x_not_y = 0.0
            for l in range(n_y):
                if l != j:
                    p_x_not_y += prob[f"P(x{i+1},y{l+1})"]

            if not (p_xy - tol <= p_yx <= 1.0 - p_x_not_y + tol):
                return False
    return True

def generate_prob_with_constraint(scm_type, n_x, n_y=None, n_z=3, max_tries=10000, tol=1e-9):
    """
    在三种结构的基础上，用拒绝采样强制满足：
        P(x_i,y_j) <= P(y_j_x_i) <= 1 - P(x_i, not y_j)
    """

    if n_y is None:
        n_y = n_x

    for _ in range(max_tries):
        if scm_type == "mediator":
            prob = generate_prob_mediator(n_x=n_x, n_y=n_y, n_z=n_z)
        elif scm_type == "confounder":
            prob = generate_prob_confounder(n_x=n_x, n_y=n_y, n_z=n_z)
        elif scm_type == "complex":
            prob = generate_prob_complex(n_x=n_x, n_y=n_y, n_z=n_z)
        else:
            raise ValueError(
                f"未知 scm_type: {scm_type}, 只能是 "
                f"'mediator', 'confounder', 'complex'"
            )

        if check_px_y_ineq(prob, n_x, n_y, tol=tol):
            return prob

    raise RuntimeError(
        f"在 {max_tries} 次随机采样中，没有找到满足 "
        f"P(x,y)<=P(y_x)<=1-P(x,not y) 的 prob（scm_type={scm_type}）"
    )
