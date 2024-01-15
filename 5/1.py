import math
from scipy.optimize import minimize_scalar

fun = lambda x: x ** 2 - math.sin(x)
fun2 = lambda x: math.exp(-x) + x**2

y = minimize_scalar(fun2, bounds=(-10, 10), method='bounded')
print(f"{y.x:.4f}")