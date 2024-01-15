import numpy as np
import math


def print_matrix(val, num):
    n, m = val.shape[0], val.shape[1]
    for i in range(n):
        for j in range(m):
            str_format = '%.' + str(num) + 'f'
            if j < m - 1:
                print(str_format % val[i, j], end=' ')
            else:
                print(str_format % val[i, j])


def print_vector(val, num):
    n = val.shape[0]
    for i in range(n):
        str_format = '%.' + str(num) + 'f'
        if i < n - 1:
            print(str_format % val[i], end=' ')
        else:
            print(str_format % val[i])


def wolfe(fun, gfun, xk, dk, rho=0.1, sig=0.5, maxiter=20, inf=1e9):
    a, b = 0, inf
    alpha = 1
    fk = fun(xk)
    gk = gfun(xk)
    t = gk.T @ dk
    for k in range(maxiter):
        xk1 = xk + alpha * dk
        fk1 = fun(xk1)
        gk1 = gfun(xk1)
        t1 = gk1.T @ dk
        a1 = fk - fk1
        a2 = -rho * alpha * t
        a3 = sig * t
        if a1 >= a2 and t1 >= a3:
            break
        if a1 >= a2 and t1 < a3:
            a = alpha
            alpha = min(2 * alpha, (alpha + b)/2)
        elif a1 < a2:
            b = alpha
            alpha = (alpha + a) / 2
    return alpha


def quasi_newton(fun, x0, gfun, epsilon=1e-6, maxiter=100, method='sr1', verb=True):
    n = len(x0)
    xk = x0
    k = 0
    Hk = np.eye(n)
    for k in range(maxiter):
        gk = gfun(xk)
        gk2 = np.linalg.norm(gk)
        if verb is True:
            print(k, xk, fun(xk), gk2)
        if gk2 < epsilon:
            break
        try:
            dk = - Hk @ gk
        except np.linalg.LinAlgError as result:
            print('error: %s' % result)
            return xk, fun(xk)
        alpha = wolfe(fun, gfun, xk, dk)
        xk = xk + alpha * dk
        if method == 'sr1':
            sk = alpha * dk
            yk = gfun(xk) - gk
            sr1 = sk - Hk @ yk
            Hk = Hk + sr1 @ sr1.T / (sr1.T @ yk)
        if method == 'dfp':
            sk = alpha * dk
            yk = gfun(xk) - gk
            Hk = Hk + sk @ sk.T / (sk.T @ yk) - Hk @ yk @ yk.T @ Hk / (yk.T @ Hk @ yk)
        if method == 'bfgs':
            sk = alpha * dk
            yk = gfun(xk) - gk
            Hk = Hk + (1 + yk.T @ Hk @ yk / (yk.T @ sk)) * (sk @ sk.T) / (yk.T @ sk) - (sk @ yk.T @ Hk + Hk @ yk @ sk.T) / (yk.T @ sk)
    return xk, fun(xk), k


# Lagrange function
def mpsi(x, fun, hf, gf, dfun, dhf, dgf, mu, lam, sigma):
    f, he, gi = fun(x), hf(x), gf(x)
    l, m = he.shape[1], gi.shape[1]  # 等式约束和不等式约束个数
    psi = f  # （7.41）
    s1 = 0.0
    # 等式约束（7.41）
    for i in range(0, l):
        psi = psi - he[0][i] * mu[i]
        s1 = s1 + he[0][i] ** 2
    psi = psi + 0.5 * sigma * s1
    s2 = 0.0
    # 不等式约束（7.41）
    for i in range(0, m):
        s3 = np.minimum(0.0, gi[0][i] - lam[i] / sigma)
        s2 = s2 + s3 ** 2 - (lam[i] / sigma) ** 2
    psi = psi + s2 * sigma * 0.5
    return psi


# gradient for Lagrange function
def dmpsi(x, fun, hf, gf, dfun, dhf, dgf, mu, lam, sigma):
    dpsi = dfun(x)
    he, gi = hf(x), gf(x)
    dhe, dgi = dhf(x), dgf(x)
    l, m = he.shape[1], gi.shape[1]  # 等式约束和不等式约束个数
    for i in range(0, l):  # 等式约束梯度
        dpsi = dpsi + (sigma * he[0][i] - mu[i]) * dhe[:, i, None]
    for i in range(0, m):  # 不等式约束梯度
        dpsi = dpsi + min(sigma * gi[0][i] - lam[i], 0.0) * dgi[:, i, None]
    return dpsi


def multphr(fun, hf, gf, dfun, dhf, dgf, x0, sigma=2.0, rho=2.0, theta=0.8, epsilon=1e-6, maxiter=500, verb=True):
    # hf, gf: equation and inequation
    x = x0
    he, gi = hf(x), gf(x)
    n, l, m = x.shape[0], he.shape[1], gi.shape[1]  # 目标变量、等式约束和不等式约束个数
    # Lagrange Multiplier
    mu, lam = 0.1 * np.ones(shape=(l,)), 0.1 * np.ones(shape=(m,))
    k, ink = 0, 0  # 外部和内部迭代次数
    btak, btaold = 10, 10  # 停止条件
    for k in range(0, maxiter):
        if verb is True:
            print('k=%d, f(x)=%.4f, ||c(x)||=%.9f' % (k, fun(x), btak))
            print_vector(x, 4)
        mpsi_fun = lambda x: mpsi(x, fun, hf, gf, dfun, dhf, dgf, mu, lam, sigma)
        dmpsi_fun = lambda x: dmpsi(x, fun, hf, gf, dfun, dhf, dgf, mu, lam, sigma)
        # 求解子问题
        x, fk, ik = quasi_newton(mpsi_fun, x0, dmpsi_fun, method='bfgs', verb=False)
        ink = ink + ik
        he, gi = hf(x), gf(x)
        btak = 0.0  # 等式和不等式约束L2范数
        for i in range(0, l):
            btak = btak + he[0][i] ** 2
        for i in range(0, m):
            temp = np.minimum(gi[0][i], lam[i] / sigma)
            btak = btak + temp ** 2
        btak = np.sqrt(btak)
        if btak > epsilon:
            if k >= 2 and btak > theta * btaold:
                sigma = rho * sigma  # 放大sigma
            # update multiplier
            for i in range(0, l):
                mu[i] = mu[i] - sigma * he[0][i]
            for i in range(0, m):
                lam[i] = -sigma * np.minimum(0.0, gi[0][i] - lam[i] / sigma)
        else:
            break
        btaold = btak
        x0 = x
        pass
    return x, mu, lam, (fun(x), k, ink, btak)
    pass


if __name__ == '__main__':
    # n：变量个数，l：等式个数，m：不等式个数

    # fun: 目标函数
    fun = lambda x: (x[0][0] - 2.0) ** 2 + (x[1][0] - 1.0) ** 2

    # hf: 等式约束
    hf = lambda x: np.array([[x[0][0] - 2.0 * x[1][0] + 1.0]])  # 1 x l

    # gf: 不等式约束
    gf = lambda x: np.array([[-0.25 * x[0][0] ** 2 - x[1][0] ** 2 + 1.0]])  # 1 x m
    
    dfun = lambda x: np.array([[2.0 * (x[0][0] - 2.0)], [2.0 * (x[1][0] - 1.0)]])  # n x 1
    dhf = lambda x: np.array([[1.0], [-2.0]])  # n x l
    dgf = lambda x: np.array([[-0.5 * x[0][0]], [-2.0 * x[1][0]]])  # n x m
    
    x0 = np.array([[3.0], [3.0]])  # n x 1
    
    x, mu, lam, info = multphr(fun, hf, gf, dfun, dhf, dgf, x0)
    print(x.tolist(), mu, lam, info)
    pass
