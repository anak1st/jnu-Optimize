from scipy.optimize import linprog

# c = [-1, 4]
# A = [[-3, 1], [1, 2]]
# b = [6, 4]
# x0_bounds = (None, None)
# x1_bounds = (-3, None)
# linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])

c = [-4, -1]
A = [[-1, 2],
     [2, 3],
     [1, -1]]
b = [4, 12, 3]
x0_bounds = (0, None)
x1_bounds = (0, None)
y = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
for i in range(2):
    print(f"{y.x[i]:.4f}", end=" ")