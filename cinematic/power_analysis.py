import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from statistics import mean


def F(fi):
    if (-40 / 180 * math.pi + 2 * math.pi) <= fi < (40 / 180 * math.pi + 2 * math.pi):
        return 40
    else:
        return 0


x, l1, l2, l3, x_d, y_d, x_a, y_a, a, b, c, ad, db, x_c, x_b, y_c, y_b, l4 = \
    sp.symbols('x l1 l2 l3 x_d y_d x_a y_a a b c ad db x_c x_b y_c y_b l4')
l1_value = 1
l2_value = 2
l3_value = 2
l4_value = 3

x_d_value = 1
y_d_value = 1

x_a_value = 2
y_a_value = 3

x_k_value = -5
y_k_value = 5

ad_value = np.sqrt((x_d_value - x_a_value) ** 2 + (y_d_value - y_a_value) ** 2)

c_valu = "pi / 2 - atan((abs(x_d - x_a)) / abs(y_d - y_a))"
c_value_func = sp.simplify(c_valu)
c_value = c_value_func.subs({x_d: x_d_value, x_a: x_a_value, y_d: y_d_value, y_a: y_a_value})
c_value_degrees = math.degrees(c_value.evalf())
c_value = c_value.evalf()


all_solve = []
for fi in np.arange(-40 / 180 * math.pi + 2 * math.pi, 40 / 180 * math.pi + 2 * math.pi, 0.001):
    b_x = "x_a + l1 * cos(x)"
    b_y = "y_a - l1 * sin(x)"
    b_x_func = sp.sympify(b_x)
    b_y_func = sp.sympify(b_y)
    b_x_evaluated = b_x_func.subs({l1: l1_value, x_a: x_a_value})
    b_y_evaluated = b_y_func.subs({l1: l1_value, y_a: y_a_value})

    db_value = (b_x_evaluated - 1) ** 2 + (b_y_evaluated - 1) ** 2

    angle = math.degrees(fi)
    c_x = "x_d + l3 * cos(b + c - a)" if (
                (angle > (360 - c_value_degrees)) or (angle < (180 - c_value_degrees))) else "x_d + l3 * cos(b + c + a)"
    c_y = "y_d + l3 * sin(b + c - a)" if (
                (angle > (360 - c_value_degrees)) or (angle < (180 - c_value_degrees))) else "y_d + l3 * sin(b + c + a)"
    c_x_func = sp.sympify(c_x)
    c_y_func = sp.sympify(c_y)

    a_value = "acos((db - l1**2 + ad**2) / (2 * ad * db**0.5))"
    b_value = "acos((db - l2**2 + l3**2) / (2 * l3 * db**0.5))"
    a_value_func = sp.sympify(a_value)
    b_value_func = sp.sympify(b_value)
    b_value_evaluated = b_value_func.subs({db: db_value, l2: l2_value, l3: l3_value, ad: ad_value})
    a_value_evaluated = a_value_func.subs({db: db_value, l1: l1_value, ad: ad_value})

    c_x_evaluated = c_x_func.subs(
        {l3: l3_value, x_d: x_d_value, b: b_value_evaluated, c: c_value, a: a_value_evaluated})
    c_y_evaluated = c_y_func.subs(
        {l3: l3_value, y_d: y_d_value, b: b_value_evaluated, c: c_value, a: a_value_evaluated})

    e_x = "x_c + ((l4 * (x_c - x_b)) / (l2))"
    e_y = "y_c + ((l4 * (y_c - y_b)) / (l2))"
    e_x_func = sp.sympify(e_x)
    e_y_func = sp.sympify(e_y)
    e_x_evaluated = e_x_func.subs({l2: l2_value, l4: l4_value, x_c: c_x_evaluated, x_b: b_x_evaluated})
    e_y_evaluated = e_y_func.subs({l2: l2_value, l4: l4_value, y_c: c_y_evaluated, y_b: b_y_evaluated})

    b_x_rounded = b_x_evaluated.subs(x, fi).evalf(50)
    b_y_rounded = b_y_evaluated.subs(x, fi).evalf(50)
    c_x_rounded = c_x_evaluated.subs(x, fi).evalf(50)
    c_y_rounded = c_y_evaluated.subs(x, fi).evalf(50)
    e_x_rounded = e_x_evaluated.subs(x, fi).evalf(50)
    e_y_rounded = e_y_evaluated.subs(x, fi).evalf(50)

    B = np.array([0., 0., 0., 0., -F(fi), -F(fi) * abs(x_k_value - e_x_rounded), 0., 0., 0.], dtype=np.float64)
    A = np.array([[1., 0., 1., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 1., 0., 0., 0., 0., 0.],
                  [abs(y_k_value - y_a_value), abs(x_k_value - x_a_value), abs(y_k_value - b_y_rounded),
                   abs(x_k_value - b_x_rounded), 0., 0., 0., 0., -1.],
                  [0., 0., -1., 0., 1., 0., 0., 0., 0.],
                  [0., 0., 0., -1., 0., 1., 0., 0., 0.],
                  [0., 0., -abs(y_k_value - b_y_rounded), -abs(x_k_value - b_x_rounded),
                   abs(y_k_value - c_y_rounded), abs(x_k_value - c_x_rounded), 0., 0., 0.],
                  [0., 0., 0., 0., -1., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 0., -1., 0., 1., 0.],
                  [0., 0., 0., 0., -abs(y_k_value - c_y_rounded), -abs(x_k_value - c_x_rounded),
                   abs(y_k_value - y_d_value), abs(x_k_value - x_d_value), 0.]], dtype=np.float64)
    if fi == -40 / 180 * math.pi + 2 * math.pi:
        D = np.array([0., 0., 0., 0., -F(fi), -F(fi) * abs(x_k_value - e_x_rounded), 0., 0., 0.])
        C = np.array([[1., 0., 1., 0., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 1., 0., 0., 0., 0., 0.],
                      [abs(y_k_value - y_a_value), abs(x_k_value - x_a_value), abs(y_k_value - b_y_rounded),
                       abs(x_k_value - b_x_rounded), 0., 0., 0., 0., -1.],
                      [0., 0., -1., 0., 1., 0., 0., 0., 0.],
                      [0., 0., 0., -1., 0., 1., 0., 0., 0.],
                      [0., 0., -abs(y_k_value - b_y_rounded), -abs(x_k_value - b_x_rounded),
                       abs(y_k_value - c_y_rounded), abs(x_k_value - c_x_rounded), 0., 0., 0.],
                      [0., 0., 0., 0., -1., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 0., -1., 0., 1., 0.],
                      [0., 0., 0., 0., -abs(y_k_value - c_y_rounded), -abs(x_k_value - c_x_rounded),
                       abs(y_k_value - y_d_value), abs(x_k_value - x_d_value), 0.]])
    solve = np.linalg.solve(A, B)
    all_solve.append(solve)
print(all_solve[0])
all_solve = np.transpose(all_solve)
R = ["Rx10", "Ry10", "Rx12", "Ry12", "Rx23", "Ry23", "Rx30", "Ry30", "M1"]
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    print(all_solve[i])
    plt.plot(np.arange(-40 / 180 * math.pi + 2 * math.pi, 40 / 180 * math.pi + 2 * math.pi, 0.001), all_solve[i])
    plt.xlabel('fi')
    plt.ylabel(R[i])
    plt.grid(True)
plt.tight_layout()

plt.figure()
for i in range(9):
    plt.plot(np.arange(-40 / 180 * math.pi + 2 * math.pi, 40 / 180 * math.pi + 2 * math.pi, 0.001), all_solve[i], label=R[i])
    plt.xlabel('fi')
    plt.grid(True)
plt.legend()
plt.show()

'''all_solve2 = []
w = 2.094
m1 = 0.026
m2 = 0.151
m3 = 0.104
g = 9.8

for fi in np.arange(50 / 180 * math.pi, 130 / 180 * math.pi, 0.001):
    f1x = AB / 2 * np.cos(fi) * w ** 2
    f1y = AB / 2 * np.sin(fi) * w ** 2
    g1 = m1 * g
    f2x = BD / 2 * np.cos(fi2(fi) - math.pi) * w ** 2
    f2y = BD / 2 * np.sin(fi2(fi) - math.pi) * w ** 2
    g2 = m2 * g
    B = np.array([-f1x, -f1y, - AB / 2 * np.sin(fi) * f1x - AB / 2 *
                  np.cos(fi) * (f1y - g1), -F(fi) * np.cos(fi2(fi)) - f2x, -F(fi) *
                  np.sin(fi2(fi)) - f2y, -BD * F(fi) - BD / 2 * np.sin(fi2(fi)) * f2x - BD /
                  2 * np.cos(fi2(fi) * (f2y - g2)),
                  0., 0., 0.])
    A = np.array([[1., 0., 1., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., AB * np.sin(fi), AB * np.cos(fi), 0., 0., 0., 1., 0.],
                  [0., 0., -1., 0., 1., 0., 0., 0., 0.],
                  [0., 0., 0., -1., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 0., -BC * np.sin(fi2(fi)), -BC * np.cos(fi2(fi)), 0., 0., 0.],
                  [0., 0., 0., 0., -1., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 0., -1., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    solve = np.linalg.solve(A, B)
    all_solve2.append(solve)
print(all_solve2[0])
all_solve2 = np.transpose(all_solve2)
R = ["Rx01", "Ry01", "Rx12", "Ry12", "Rx23", "Ry23", "Rn30", "M", "M30"]
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.plot(np.arange(50 / 180 * math.pi, 130 / 180 * math.pi, 0.001), all_solve[i])
    plt.xlabel('fi')
    plt.ylabel(R[i])
    plt.grid(True)
plt.tight_layout()
rel_error = []
rel_average = []
for i in range(9):
    rel_error.append(abs((all_solve[i] - all_solve2[i]) / all_solve2[i] * 100))
    rel_average.append(mean(rel_error[-1]))
print(rel_average)
plt.show()'''
