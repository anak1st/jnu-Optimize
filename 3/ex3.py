import numpy as np
import math
import matplotlib.pyplot as plt
# np.seterr(all='ignore')
# import warnings
# warnings.filterwarnings("ignore")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch import autograd


def jacobian(fun, x, delta=1e-3):
    g = x.copy()
    n = len(x)
    delta = np.minimum(delta, delta * np.abs(x))
    for i in range(n):
        t1, t0 = x.copy(), x.copy()
        t1[i], t0[i] = t1[i] + delta[i], t0[i] - delta[i]
        g[i] = (fun(t1) - fun(t0)) / (2 * delta[i])
    return g


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


def quasi_newton(fun, x0, gfun, epsilon=1e-6, maxiter=100, method='sr1', verb=True):
    n = len(x0)
    xk = x0
    k = 0
    Hk = np.eye(n)
    for k in range(maxiter):
        gk = gfun(xk)
        gk2 = np.linalg.norm(gk)
        if verb is True:
            print(k, xk.tolist(), fun(xk), gk2)
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


def loss(x, p):
    n, m = len(p) // 3, len(x)
    w, v, e = p[0: n], p[n: 2 * n], p[2 * n: 3 * n]
    val = 0.0
    for i in range(m):
        t = 0.0
        for j in range(n):
            ex = math.exp(e[j] - w[j] * x[i])
            t += (2 - 1/x[i]) * v[j] / (1 + ex)
            t += (x[i]-1) * v[j] * w[j] * ex / (1 + ex) ** 2
        val += (-x[i]**2 + 0.5/x[i] + t) ** 2
    return val


def output(x, p):
    n, m = len(p) // 3, len(x)
    w, v, e = p[0: n], p[n: 2 * n], p[2 * n: 3 * n]
    val = np.zeros(shape=(m,), dtype=np.float64)
    for j in range(n):
        val += v[j] / (1 + np.exp(e[j] - w[j] * x))
    return val


def grad(x, p):
    n, m = len(p) // 3, len(x)
    w, v, e = p[0: n], p[n: 2 * n], p[2 * n: 3 * n]
    val = np.zeros(shape=(3 * n, 1), dtype=np.float64)
    for i in range(m):
        t = 0.0
        for j in range(n):
            ex = math.exp(e[j] - w[j] * x[i])
            t += (2 - 1/x[i]) * v[j] / (1 + ex)
            t += (x[i]-1) * v[j] * w[j] * ex / (1 + ex) ** 2
        t = 2 * (-x[i]**2 + 0.5/x[i] + t)
        for j in range(0, n):
            tmp = 0.0
            ex = math.exp(e[j] - w[j] * x[i])
            tmp += (2 - 1/x[i]) * v[j] * x[i] * ex / (1 + ex) ** 2
            tmp += (x[i]-1) * v[j] * ((ex - w[j] * ex * x[i]) * (1 + ex) ** 2 + 2 * w[j] * ex * (1 + ex) * ex * x[i]) / (1 + ex) ** 4
            val[j] += t * tmp
        for j in range(0, n):
            tmp = 0.0
            ex = math.exp(e[j] - w[j] * x[i])
            tmp += (2 - 1/x[i]) / (1 + ex)
            tmp += (x[i]-1) * w[j] * ex / (1 + ex) ** 2
            val[j + n] += t * tmp
        for j in range(0, n):
            tmp = 0.0
            ex = math.exp(e[j] - w[j] * x[i])
            tmp += - (2 - 1/x[i]) * v[j] * ex / (1 + ex) ** 2
            tmp += (x[i]-1) * v[j] * w[j] * (ex * (1 + ex) ** 2 - 2 * ex * (1 + ex) * ex) / (1 + ex) ** 4
            val[j + 2 * n] += t * tmp
    return val


# class DiffNet(nn.Module):
#     def __init__(self, n):
#         super().__init__()
#         w = torch.ones(n)  # n,
#         v = torch.ones(n)
#         e = torch.ones(n)
#         self.w = nn.Parameter(w)
#         self.v = nn.Parameter(v)
#         self.e = nn.Parameter(e)
#         self.phi = nn.Sigmoid()

#     def forward(self, x):
#         # x: m x 1
#         z = x * self.w - self.e  # m x n
#         z = self.phi(z)
#         y = 1 + x * torch.sum(z, dim=-1, keepdim=True)
#         f = x - y + 1
#         dydx = autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
#         loss = torch.sum((dydx - f) ** 2)
#         return loss, y, dydx


# class Diff:
#     def __init__(self):
#         super().__init__()
#         self.n = 5
#         self.m = 15
#         self.lr = 1e-2
#         self.lr_fun = lambda epoch: 1.0
#         self.model = DiffNet(self.n).cuda()
#         self.opt = optim.SGD(self.model.parameters(), lr=self.lr)
#         self.scheduler = optim.lr_scheduler.LambdaLR(self.opt, self.lr_fun)
#         pass

#     def train(self):
#         xi = np.linspace(0, 1, self.m, dtype=np.float32).reshape(self.m, 1)
#         xi = torch.tensor(xi, requires_grad=True).cuda()
#         for epoch in range(5000):
#             lr = self.opt.param_groups[0]['lr']
#             loss, y, dy = self.model(xi)
#             # self.dydx(dy, xi)
#             self.opt.zero_grad()
#             loss.backward()
#             self.opt.step()
#             self.scheduler.step()
#             if (epoch + 1) % 100 == 0:
#                 print(epoch, lr, loss)
#         # show
#         _, y, _ = self.model(xi)
#         xi = xi.cpu().detach().numpy()
#         yfun = lambda x: x + np.exp(-x)
#         y = y.cpu().detach().numpy()
#         plt.figure()
#         plt.plot(xi, yfun(xi), '-b')
#         plt.plot(xi, y, '--r')
#         plt.show()

#     def dydx(self, dy, x):
#         dy = dy.cpu().detach().numpy()
#         x = x.cpu().detach().numpy()
#         w = self.model.w.cpu().detach().numpy()
#         v = self.model.v.cpu().detach().numpy()
#         e = self.model.e.cpu().detach().numpy()
#         val = np.zeros([self.m, 1])
#         for i in range(self.m):
#             t = 0.0
#             for j in range(self.n):
#                 ex = math.exp(e[j] - w[j] * x[i])
#                 t += v[j] / (1 + ex)
#                 t += x[i] * v[j] * w[j] * ex / (1 + ex) ** 2
#             val[i] = t
#         print(np.max(np.abs(dy - val)))


if __name__ == '__main__':
    n, m = 2, 15  # 2,3; 2,5; 2,10; 3,3; 3,5; 3,10; 5,15
    xi = np.linspace(1, 2, m, dtype=np.float64)
    fun = lambda p: loss(xi, p)
    gfun = lambda p: grad(xi, p)
    gf = lambda p: jacobian(fun, p).reshape(3 * n, 1)
    
    method = 'bfgs'
    p0 = np.ones(shape=(3 * n, 1), dtype=np.float64)
    # print(gfun(p0).tolist())
    # print(gf(p0).tolist())
    # print('--------------------------------------------------')
    pk, _, _ = quasi_newton(fun, p0, gfun, epsilon=1e-5, method=method, verb=False)
    
    yfun = lambda x: 0.25 * x ** 3 + 0.25 / x
    yp = lambda x, p: 0.5 + (x - 1) * output(x, p)
    # print('--------------------------------------------------')
    # print(yfun(xi))
    # print(yp(xi, pk))

    # x = float(input())
    # x = np.array([x], dtype=np.float64)
    # y = yp(x, pk)
    # print(f"{y[0]:.2f}")
    
    xi = np.linspace(1, 2, 100, dtype=np.float64)
    plt.figure()
    plt.plot(xi, yfun(xi), '-b')
    plt.plot(xi, yp(xi, pk), '--r')
    plt.show()

    # net = Diff()
    # net.train()



