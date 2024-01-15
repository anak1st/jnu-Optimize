import numpy as np
import math
np.seterr(all='ignore')
import warnings
warnings.filterwarnings("ignore")


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
            print(str_format % val[i], end=', ')
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


def armijo(fun, gfun, xk, dk, beta=0.55, rho=0.4, maxiter=20):
    m = 0
    for k in range(maxiter):
        fk1 = fun(xk + beta ** m * dk)
        fk0 = fun(xk) + gfun(xk).T @ dk * rho * beta**m
        if fk1 <= fk0:
            break
        m = m + 1
    alpha = beta ** m
    return alpha


def cg(fun, gfun, x0, G, epsilon=1e-6, maxiter=1000, verb=True):
    g0 = gfun(x0)
    d0 = -g0
    k = 0
    for k in range(maxiter):
        g = gfun(x0)
        g2 = np.linalg.norm(g)
        if verb is True:
            # print('k=%d, f(x0)=%.4f, ||d||=%.9f' % (k, fun(x0), g2))
            print_vector(x0, 5)
        if g2 < epsilon:
            break
        if k == 0:
            d = -g
        else:
            beta = g.T @ g / (g0.T @ g0)
            d = -g + beta * d0
        alpha = - d.T @ g / (d.T @ G @ d)  # line search
        x0 = x0 + alpha * d
        g0 = g
        d0 = d
    return x0, fun(x0), k


def frcg(fun, gfun, x0, epsilon=1e-5, maxiter=5000, verb=True):
    n = len(x0)
    g0 = gfun(x0)
    d0 = -g0
    k = 0
    for k in range(maxiter):
        g = gfun(x0)
        g2 = np.linalg.norm(g)
        if verb is True:
            print('k=%d, f(x0)=%.4f, ||d||=%.9f' % (k, fun(x0), g2))
            print_vector(x0, 4)
        if g2 < epsilon:
            break
        # restart
        itern = k - (n + 1) * math.floor(k / (n + 1))
        itern = itern + 1
        if itern == 1:
            d = -g
        else:
            beta = g.T @ g / (g0.T @ g0)
            d = -g + beta * d0
            gd = g.T @ d
            if gd >= 0.0:
                d = -g
        alpha = wolfe(fun, gfun, x0, d)
        x0 = x0 + alpha * d
        g0 = g
        d0 = d
    return x0, fun(x0), k


if __name__ == '__main__':
    # 采用共轭梯度法求解线性方程组
    # 
    # ex1
    G = list(map(float, input().split()))
    b = list(map(float, input().split()))

    n = len(b)

    G = np.array(G).reshape(n, n)
    b = np.array(b).reshape(n, 1)

    n = len(b)
    
    c = 0

    # print('===================================================')
    fun = lambda x: 0.5 * (x.T @ G @ x) + b.T @ x + c
    gfun = lambda x: G @ x + b
    x0 = np.zeros(n).reshape(n, 1)
    xk, fk, k = cg(fun, gfun, x0, G)
    # print('--------------------------------------------------')
    # xk, fk, k = frcg(fun, gfun, x0)

    # print('===================================================')
    # G = np.array([21, 4, 4, 1])
    # G = np.reshape(G, newshape=(2, 2))
    # fun = lambda x: 0.5 * (x.T @ G @ x) + b.T @ x + c
    # gfun = lambda x: G @ x + b
    # x0 = np.array([[-30], [100]])
    # xk, fk, k = cg(fun, gfun, x0, G)
    # print('--------------------------------------------------')
    # xk, fk, k = frcg(fun, gfun, x0)

    # ex2
    # fun = lambda x: 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2
    # gfun = lambda x: np.array([400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1), -200 * (x[0] ** 2 - x[1])]).reshape(2, 1)

    # print('===================================================')
    # x0 = np.array([0.0, 0.0]).reshape(2, 1)
    # xk, fk, k = frcg(fun, gfun, x0)
    # print('===================================================')
    # x0 = np.array([0.5, 0.5]).reshape(2, 1)
    # xk, fk, k = frcg(fun, gfun, x0)



