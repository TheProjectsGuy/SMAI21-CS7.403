# %% Import everything
import numpy as np
import sympy as sp
from IPython.display import display

# %%
M = np.array([
    [4, 8],
    [11, 7],
    [14, -2]
])
print(f"M = {M}")
M_L = M @ M.T
M_R = M.T @ M

# %% Left singular values
l = sp.symbols(r'\lambda')
i3 = sp.Matrix(np.eye(3))
M_L_sp = sp.Matrix(M_L)
eq = M_L_sp - l * i3
eqd = eq.det()
print("Determinant is")
display(eqd)
# Solutions
lvals = sp.solve(sp.Eq(eqd, 0), l)
print(f"Lambda values are: {lvals}")
# Eigen-vectors
x, y, z = sp.symbols('x, y, z')
v = sp.Matrix([[x], [y], [z]])
print(f"Solving eigenvectors for lambda = {lvals[0]}")
res = sp.solve(sp.Eq(M_L_sp * v, lvals[0] * v), x, y, z)
print(f"\t{res}")
print(f"Solving eigenvectors for lambda = {lvals[1]}")
res = sp.solve(sp.Eq(M_L_sp * v, lvals[1] * v), x, y, z)
print(f"\t{res}")
print(f"Solving eigenvectors for lambda = {lvals[2]}")
res = sp.solve(sp.Eq(M_L_sp * v, lvals[2] * v), x, y, z)
print(f"\t{res}")

# %% Right singular values
M_R_sp = sp.Matrix(M_R)
i2 = sp.Matrix(np.eye(2))
eq = M_R_sp - l * i2
eqd = eq.det()
print("Determinant is")
display(eqd)
# Solutions
lvals = sp.solve(sp.Eq(eqd, 0), l)
print(f"Lambda values are: {lvals}")
# Eigen-vectors
x, y = sp.symbols('x, y')
v = sp.Matrix([[x], [y]])
print(f"Solving eigenvectors for lambda = {lvals[0]}")
res = sp.solve(sp.Eq(M_R_sp * v, lvals[0] * v), x, y)
print(f"\t{res}")
print(f"Solving eigenvectors for lambda = {lvals[1]}")
res = sp.solve(sp.Eq(M_R_sp * v, lvals[1] * v), x, y)
print(f"\t{res}")

# %% Singular values
z1, z2, z3 = sp.symbols("z_1, z_2, z_3")
y1, y2 = sp.symbols("y_1, y_2")
s1, s2 = sp.symbols(r"\sigma_1, \sigma_2")
U = sp.Matrix([
    [0.5*z1, -z2, 2*z3],
    [z1, -0.5*z2, -2*z3],
    [z1, z2, z3]
])
S = sp.Matrix([
    [s1, 0],
    [0, s2],
    [0, 0]
])
V = sp.Matrix([
    [3*y1, -y2/3],
    [y1, y2]
])
sub_values = {
    z1: 2/3,
    z2: -2/3,   # Positive singular values only!
    z3: 1/3,
    y1: 1/((10)**0.5),
    y2: 3/((10)**0.5)
}
M_sp = sp.Matrix(M)
M_esp = U * S * V.T
M_svd_rhs = M_esp.subs(sub_values)
s_sols = sp.solve(sp.Eq(M_sp, M_svd_rhs), s1, s2)
display(s_sols)
M_another = M_esp.subs(sub_values).subs(s_sols)
print("U")
display(U.subs(sub_values))
print("S (singular matrix)")
display(S.subs(s_sols))
print("V*")
display(V.subs(sub_values).T)
print("The final M (reconstructed) is")
display(M_another)

# %% Actual SVD results
u, s, vh = np.linalg.svd(M)
print(f"Left-singular vectors \nU = {u}")
print(f"Right-singular values' conjugate transpose \nV* = {vh}")
print(f"Singular values \nS = {s}")

# %%
