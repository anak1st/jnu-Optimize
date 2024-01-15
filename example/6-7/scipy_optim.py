import math
import numpy as np
from scipy.optimize import line_search, minimize_scalar, minimize, least_squares, leastsq, root, linprog, \
    rosen, rosen_der, rosen_hess


def wolfe():
    print('======================= wolfe ============================')
    fun = lambda x: 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2
    gfun = lambda x: np.array([400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1), -200 * (x[0] ** 2 - x[1])])
    xk = np.array([-1, 1])
    dk = np.array([1, -2])
    res = line_search(fun, gfun, xk, dk)
    print(res[0])


def accurate_line_search():
    print('======================= accurate line search ============================')
    fun = lambda x: x ** 2 - math.sin(x)
    res = minimize_scalar(fun, bounds=(0, 1), method='bounded')
    print(res.x)


def Newton():
    print('======================= Newton ============================')
    x0 = np.zeros(5)
    res = minimize(rosen, x0, method='Newton-CG',
                   jac=rosen_der, hess=rosen_hess,
                   options={'xtol': 1e-8, 'disp': True})
    print(res.x)


def BFGS():
    print('======================= BFGS ============================')
    x0 = np.zeros(5)
    res = minimize(rosen, x0, method='BFGS',
                   jac=rosen_der,
                   options={'gtol': 1e-6, 'disp': True})
    print(res.x)


def CG():
    print('======================= CG ============================')
    b = np.array([2, 3])
    c = 10
    G = np.array([21, 4, 4, 15])
    G = np.reshape(G, newshape=(2, 2))
    fun = lambda x: 0.5 * (x.T @ G @ x) + b.T @ x + c
    gfun = lambda x: G @ x + b
    x0 = np.array([-30, 100])
    res = minimize(fun, x0, method='CG',
                   jac=gfun,
                   options={'gtol': 1e-6, 'disp': True})
    print(res.x)


def least_square():
    print('======================= least square (bound) ============================')
    Rfun = lambda x: np.array([x[0] - 0.7 * np.sin(x[0]) - 0.2 * np.cos(x[1]),
                               x[1] - 0.7 * np.cos(x[0]) + 0.2 * np.sin(x[1])])
    Jfun = lambda x: np.array([1 - 0.7 * np.cos(x[0]), 0.2 * np.sin(x[1]),
                               0.7 * np.sin(x[0]), 1 + 0.2 * np.cos(x[1])]).reshape(2, 2)
    fun = lambda x: 0.5 * Rfun(x).T @ Rfun(x)
    gfun = lambda x: Jfun(x).T @ Rfun(x)
    x0 = np.array([0.0, 0.0])
    res = least_squares(fun, x0, gfun, verbose=1)
    print(res.x)
    print('======================= least square ============================')
    res = leastsq(Rfun, x0, Dfun=Jfun)
    print(res[0])
    print('======================= least square (lm) ============================')
    res = root(Rfun, x0, jac=Jfun, method='lm')
    print(res.x)


def line_programming():
    print('======================= line programming ============================')
    """
    min c @ x
    s.t. 
        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub
    """
    c = [-1, 4]
    A = [[-3, 1], [1, 2]]
    b = [6, 4]
    x0_bounds = (None, None)
    x1_bounds = (-3, None)
    res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
    print(res.x, res.fun)


def constraint_optimization():
    print('======================= constrained optimization ============================')
    fun = lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2
    hf = lambda x: np.array([x[0] - 2.0 * x[1] + 1.0])
    gf = lambda x: np.array([-0.25 * x[0] ** 2 - x[1] ** 2 + 1.0])
    dfun = lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]).reshape(2, 1)
    dhf = lambda x: np.array([1.0, -2.0])
    dgf = lambda x: np.array([-0.5 * x[0], -2.0 * x[1]])
    cons = ({'type': 'eq', 'fun': hf, 'jac': dhf},
            {'type': 'ineq', 'fun': gf, 'jac': dgf})
    x0 = np.array([3.0, 3.0])
    res = minimize(fun, x0, jac=dfun, constraints=cons, method='SLSQP', options={'disp': True})
    print(res.x)


if __name__ == '__main__':
    # wolfe()
    # accurate_line_search()
    # Newton()
    # BFGS()
    # CG()
    least_square()
    # line_programming()
    # constraint_optimization()
    pass
