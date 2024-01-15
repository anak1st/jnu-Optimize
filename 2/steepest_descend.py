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


def armijo(fun, gfun, xk, dk, beta=0.5, rho=0.2, maxiter=20):
    m = 0
    for k in range(maxiter):
        fk1 = fun(xk + beta ** m * dk)
        fk0 = fun(xk) + gfun(xk).T @ dk * rho * beta**m
        if fk1 <= fk0:
            break
        m = m + 1
    alpha = beta ** m
    return alpha


def golds(fun, a, b, epsilon=1e-5, delta=1e-4, maxiter=100, verb=False):
    tau = (math.sqrt(5) - 1) / 2
    h = b - a
    fa, fb = fun(a), fun(b)
    al, ar = a + (1 - tau) * h, a + tau * h
    fal, far = fun(al), fun(ar)
    alpha = (a + b) / 2
    for k in range(maxiter):
        if fal < far:
            b, fb = ar, far
            ar, far = al, fal
            h = b - a
            al = a + (1 - tau) * h
            fal = fun(al)
        else:
            a, fa = al, fal
            al, fal = ar, far
            h = b - a
            ar = a + tau * h
            far = fun(ar)
        if verb is True:
            print(k, a, b, fa, fb)
        if math.fabs(b - a) <= epsilon and math.fabs(fb - fa) <= delta:
            alpha = (a + b) / 2
            break
    return alpha


def line_search(fun, gfun, xk, dk, method='a'):
    alpha = 0.0
    if method == 'a':  # armijo
        alpha = armijo(fun, gfun, xk, dk, beta=0.5, rho=0.4)
    if method == 'g':  # gold
        f = lambda a: fun(xk + a * dk)
        alpha = golds(f, 0, 1)
        pass
    return alpha


def grad(fun, gfun, x0, search='a', g2=None, epsilon=1e-5, maxiter=1000, verb=True):
    xs = []
    for k in range(maxiter):
        g = gfun(x0)
        d = -g
        if verb is True:
            print('k=%d, f(x0)=%.4f, ||d||=%.9f' % (k, fun(x0), np.linalg.norm(d)))
            print_vector(x0, 4)
            xs.append(x0)
        if np.linalg.norm(d) < epsilon:
            break
        if g2 is not None:
            alpha = g.T @ g / (g.T @ g2 @ g)
        else:
            alpha = line_search(fun, gfun, x0, d, method=search)
        # x0 = x0 + alpha * d
        x0 = x0 + 0.01 * d
    x = x0
    return x, np.array(xs)


if __name__ == '__main__':

    A = list(map(float, input().split()))
    A = np.array(A)
    A = np.reshape(A, newshape=(4, 2))

    b = list(map(float, input().split()))
    b = np.array(b)
    b = np.reshape(b, newshape=(4, 1))

    # print(b, A)

    fun = lambda x: 0.5 * (x.T @ A.T @ A @ x + 2 * b.T @ A @ x + b.T @ b)
    gfun = lambda x: A.T @ (A @ x + b)
    x0 = np.array([0.0, 0.0]).reshape(2, 1)
    x, _ = grad(fun, gfun, x0, search='a', verb=False)  # a

    print_matrix(x, 3)



