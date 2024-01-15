import math
import numpy as np
from scipy.optimize import minimize


if __name__ == '__main__':
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    dfun = lambda x: np.array([2 * x[0], 2 * x[1]])
    cons = ({'type': 'eq', 'fun': lambda x: x[0] - x[1] + 1,
             'jac': lambda x: [1, -1]})
    # cons = ({'type': 'ineq', 'fun': lambda x: - x[0] + x[1] - 1,
    #          'jac': lambda x: [-1, 1]})
    x0 = np.array([0.0, 0.0])
    res = minimize(fun, x0, jac=dfun, constraints=cons, method='SLSQP', options={'disp': True})
    print(res.x)
    pass
