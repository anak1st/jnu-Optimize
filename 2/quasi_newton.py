import numpy as np
import math
np.seterr(divide='ignore', invalid='ignore')

def print_matrix(val, num):
    n, m = val.shape[0], val.shape[1]
    for i in range(n):
        for j in range(m):
            str_format = '%.' + str(num) + 'f'
            if j < m - 1:
                print(str_format % val[i, j], end=' ')
            else:
                print(str_format % val[i, j])


def print_vector(val):
    n = val.shape[0]
    for i in range(n):
        print(f"{val[i][0]:.4f}", end=' ')


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


def quasi_newton(fun, x0, gfun, epsilon=1e-6, maxiter=500, method='sr1', verb=True):
    n = len(x0)
    xk = x0
    k = 0
    Hk = np.eye(n)
    for k in range(maxiter):
        gk = gfun(xk)
        gk2 = np.linalg.norm(gk)
        if verb is True:
            print('k=%d, f(xk)=%.4f, ||gk||=%.9f' % (k, fun(xk), gk2))
            print_vector(xk, 4)
            pass
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


if __name__ == '__main__':
    # ex1
    a = float(input())
    fun = lambda x: a * x[0]**2 + 4 * x[1]**2 + 9 * x[2]**2 - 2 * x[0] + 18 * x[1]
    gfun = lambda x: np.array([
        2 * a * x[0] - 2,
        8 * x[1] + 18,
        18 * x[2]
    ]).reshape(3, 1)
    method = 'sr1'
    x0 = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    xk, fk, k = quasi_newton(fun, x0, gfun, epsilon=1e-5, method=method, verb=False)
    # xk = xk.reshape(1, 3)
    print_vector(xk)



