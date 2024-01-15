import numpy as np
import math
# import matplotlib.pyplot as plt


def golds(fun, a, b, epsilon=1e-5, delta=1e-4, maxiter=100, verb=True):
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
            # print(k, a, b, fa, fb)
            pass
        if math.fabs(b - a) <= epsilon and math.fabs(fb - fa) <= delta:
            alpha = (a + b) / 2
            break
    return alpha, b - a, fb - fa


if __name__ == '__main__':
    # ex1
    fun = lambda x: 1 - x * math.exp(-(x**2))
    alpha, b_a, fb_fa = golds(fun, 0, 1, 0.001)

    print(f"{alpha:.5f}")
    print(f"{b_a:.5f}")

    # print('=======================================================')
    # fun = lambda x: 100 * (x[0]**2 - x[1])**2 + (x[0] - 1) ** 2
    # xk = np.array([-1, 1])
    # dk = np.array([1, -2])
    # FUN = lambda alpha: fun(xk + alpha * dk)
    # t = np.zeros(shape=(100,))
    # x = np.linspace(0.1, 0.3, 100)
    # for i in range(0, 100):
    #     t[i] = FUN(x[i])
    # # plt.figure(num=0)
    # # plt.plot(x, t, '--b')
    # # plt.show()
    # print(golds(FUN, 0.1, 0.3))

