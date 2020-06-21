import cvxpy as cp
import numpy as np

# Problem data.
n = 64
np.random.seed(1)
A = np.random.randn(n, n)
B = np.random.randn(n, n)

one_vec = np.ones(n)

# Construct the problem.
P = cp.Variable( (n, n) )
objective = cp.Minimize(cp.norm2(A @ P - P @ B))
constraints = [P @ one_vec == one_vec, P.T @ one_vec == one_vec, P >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", P.value )
