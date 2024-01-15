import numpy as np


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


if __name__ == '__main__':
    # ex1
    fun = lambda x: 100 * (x[0]**2 - x[1])**2 + (x[0] - 1) ** 2
    gfun = lambda x: np.array([400 * x[0] * (x[0]**2 - x[1]) + 2 * (x[0] - 1), -200 * (x[0]**2 - x[1])])
    xk = np.array([-1, 1])
    dk = np.array([1, 1])
    alpha = armijo(fun, gfun, xk, dk, 0.5, 0.1)
    print(f"{alpha:.5f}")
    print(f"{fun(xk):.5f}")
    print(f"{fun(xk + alpha * dk):.5f}")

