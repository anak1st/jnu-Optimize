import math
import numpy as np
from scipy.optimize import leastsq


# Rfun = lambda x: np.array([x[0] - 0.7 * np.sin(x[0]) - 0.2 * np.cos(x[1]),
#                            x[1] - 0.7 * np.cos(x[0]) + 0.2 * np.sin(x[1])])
# Jfun = lambda x: np.array([1 - 0.7 * np.cos(x[0]), 0.2 * np.sin(x[1]),
#                            0.7 * np.sin(x[0]), 1 + 0.2 * np.cos(x[1])]).reshape(2, 2)
# x0 = np.zeros(2)
# leastsq(Rfun, x0, Dfun=Jfun)


Rfun = lambda x: np.array([x[0]**2 + x[1]**2 + x[2]**2 - 1,
                           x[0] + x[1] + x[2] - 1,
                           x[0]**2 + x[1]**2 + (x[2]-2)**2 - 1,
                           x[0] + x[1] - x[2] + 1,
                           x[0]**3 + 3*(x[1]**2) + (5*x[2]-x[0]+1)**2-36])

Jfun = lambda x: np.array([2*x[0], 2*x[1], 2*x[2],
                           1,1,1,
                           2*x[0], 2*x[1], 2*(x[2]-2),
                           1,1,-1,
                           3*(x[0]**2)-2*(5*x[2]-x[0]+1), 6*x[1], 10*(5*x[2]-x[0]+1)]).reshape(5, 3)

x0 = np.zeros(3)
y = leastsq(Rfun, x0, Dfun=Jfun)
# print(y[0])
for i in range(3):
    print(f"{y[0][i]:.4f}", end=" ")
