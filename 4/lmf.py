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


def lmf(Rfun, x0, Jfun, epsilon=1e-6, maxiter=100, verb=True):
    fun = lambda x: 0.5 * Rfun(x).T @ Rfun(x)
    gfun = lambda x: Jfun(x).T @ Rfun(x)
    n = len(x0)
    nuk = np.linalg.norm(Rfun(x0))
    xk = x0
    for k in range(maxiter):
        fk = fun(xk)
        jk = Jfun(xk)
        gk = gfun(xk)
        gk2 = np.linalg.norm(gk)
        if verb is True:
            print('k=%d, f(x0)=%.4f, ||d||=%.9f' % (k, fun(x0), gk2))
            print_vector(xk, 4)
        if gk2 < epsilon:
            break
        try:
            # dk = - np.linalg.inv(jk.T @ jk + nuk * np.eye(n)) @ gk
            dk = - np.linalg.solve(jk.T @ jk + nuk * np.eye(n), gk)
        except np.linalg.LinAlgError as result:
            print('error: %s' % result)
            return xk, fk, k
        dfk = fk - fun(xk + dk)
        dqk = 0.5 * dk.T @ (nuk * dk - gk)
        gam = dfk / dqk
        if gam < 0.25:
            nuk = nuk * 4
        elif gam > 0.75:
            nuk = nuk / 2
        else:
            pass
        if gam > 0:
            xk = xk + dk
    return xk, fk, k


if __name__ == '__main__':
    # 采用LMF方法求解非线性最小二乘问题。

    t12 = list(map(float, input().split()))
    y = list(map(float, input().split()))
    n = len(y)

    t1 = t12[:n]
    t2 = t12[n:]
 
    
    def fun(x):
        res = np.zeros(shape=(n, 1))
        for i in range(n):
            res[i, 0] = x[0] + t1[i] * np.sin(x[1]) + t2[i] * np.sin(x[2]) - y[i]
        # print("fun", res)
        return res
    
    def fung(x):
        res = np.zeros(shape=(n, 3))
        for i in range(n):
            res[i, 0] = 1
            res[i, 1] = t1[i] * np.cos(x[1])
            res[i, 2] = t2[i] * np.cos(x[2])
        # print("fung", res)
        return res


    Rfun = lambda x: fun(x)
    Jfun = lambda x: fung(x)
      
    # Rfun = lambda x: np.array([x[0] - 0.7 * np.sin(x[0]) - 0.2 * np.cos(x[1]),
    #                           x[1] - 0.7 * np.cos(x[0]) + 0.2 * np.sin(x[1])]).reshape(2, 1)
    # Jfun = lambda x: np.array([1 - 0.7 * np.cos(x[0]), 
    #                            0.2 * np.sin(x[1]),
    #                            0.7 * np.sin(x[0]), 
    #                            1 + 0.2 * np.cos(x[1])]).reshape(2, 2)
    
    x0 = np.ones(3).reshape(3, 1)
    xk, fk, k = lmf(Rfun, x0, Jfun, verb=False)
    print_vector(xk, 5)

    pass

