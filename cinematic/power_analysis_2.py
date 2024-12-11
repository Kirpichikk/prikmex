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


x, l1, l2, l3, x_d, y_d, x_a, y_a, a, b, c, ad, db, x_c, x_b, y_c, y_b, l4, ac= \
    sp.symbols('x l1 l2 l3 x_d y_d x_a y_a a b c ad db x_c x_b y_c y_b l4 ac')
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

T = 5/3
w1 = (2 * math.pi) / T
e1 = 0

ad_value = np.sqrt((x_d_value - x_a_value) ** 2 + (y_d_value - y_a_value) ** 2)

c_valu = "pi / 2 - atan((abs(x_d - x_a)) / abs(y_d - y_a))"
c_value_func = sp.simplify(c_valu)
c_value = c_value_func.subs({x_d: x_d_value, x_a: x_a_value, y_d: y_d_value, y_a: y_a_value})
c_value_degrees = math.degrees(c_value.evalf())
c_value = c_value.evalf()

all_solve2 = []
all_solve1 = []
m1 = 0.00754
m2 = 0.0377
m3 = 0.0131
g = 9.81

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

    ac_value = (c_x_evaluated - x_a_value) ** 2 + (c_y_evaluated - y_a_value) ** 2

    e_x = "x_c + ((l4 * (x_c - x_b)) / (l2))"
    e_y = "y_c + ((l4 * (y_c - y_b)) / (l2))"
    e_x_func = sp.sympify(e_x)
    e_y_func = sp.sympify(e_y)
    e_x_evaluated = e_x_func.subs({l2: l2_value, l4: l4_value, x_c: c_x_evaluated, x_b: b_x_evaluated})
    e_y_evaluated = e_y_func.subs({l2: l2_value, l4: l4_value, y_c: c_y_evaluated, y_b: b_y_evaluated})

    fi_1_value = x
    fi_2 = "pi - x + acos((l1**2 + l2**2 - ac) / (2 * l1 * l2))"
    fi_2_func = sp.sympify(fi_2)
    fi_2_value = fi_2_func.subs({l1: l1_value, l2: l2_value, ac: ac_value})
    fi_3_value = b_value_evaluated + c_value - a_value_evaluated if ((angle > (360 - c_value_degrees)) or (
                angle < (180 - c_value_degrees))) else b_value_evaluated + c_value - a_value_evaluated

    wq2 = sp.diff(fi_2_value, fi_1_value)
    w2 = wq2 * w1
    eq2 = sp.diff(wq2, fi_1_value)
    e2 = eq2 * w1 ** 2 + wq2 * e1 if -40 / 180 * math.pi + 2 * math.pi<=fi<=2 * math.pi else (-1)*(eq2 * w1 ** 2 + wq2 * e1)
    wq3 = sp.diff(fi_3_value, fi_1_value)
    w3 = wq3 * w1
    eq3 = sp.diff(wq3, fi_1_value)
    e3 = eq3 * w1 ** 2 + wq3 * e1 if -40 / 180 * math.pi + 2 * math.pi<=fi<=2 * math.pi else (-1)*(eq3 * w1 ** 2 + wq3 * e1)

    ang = math.radians(math.degrees(fi))
    b_x_rounded = b_x_evaluated.subs(x, ang).evalf(50)
    b_y_rounded = b_y_evaluated.subs(x, ang).evalf(50)
    c_x_rounded = c_x_evaluated.subs(x, ang).evalf(50)
    c_y_rounded = c_y_evaluated.subs(x, ang).evalf(50)
    e_x_rounded = e_x_evaluated.subs(x, ang).evalf(50)
    e_y_rounded = e_y_evaluated.subs(x, ang).evalf(50)


    w2_value = w2.subs(x, ang).evalf(10) if math.degrees(fi)!= 360 or math.degrees(fi)!= 0 else 0
    e2_value = e2.subs(x, ang).evalf(10) if math.degrees(fi)!= 360 or math.degrees(fi)!= 0 else 0
    w3_value = w3.subs(x, ang).evalf(10) if math.degrees(fi)!= 360 or math.degrees(fi)!= 0 else 0
    e3_value = e3.subs(x, ang).evalf(10) if math.degrees(fi)!= 360 or math.degrees(fi)!= 0 else 0

    f1_x = -m1 * abs(((x_a_value + b_x_rounded)/2) - x_k_value) * w1 ** 2
    f1_y = -m1 * abs(y_k_value - ((y_a_value + b_y_rounded)/2)) * w1 ** 2

    f2_x = -m2 * abs(((e_x_rounded + b_x_rounded)/2) - x_k_value) * w2_value ** 2
    f2_y = -m2 * abs(y_k_value - ((e_y_rounded + b_y_rounded)/2)) * w2_value ** 2

    f3_x = -m3 * abs(x_k_value - ((c_x_rounded + x_d_value)/2)) * w3_value ** 2
    f3_y = -m3 * abs(y_k_value - ((c_y_rounded + y_d_value)/2)) * w3_value ** 2

    g1 = g * m1
    g2 = g * m2
    g3 = g * m3

    i2 = (m2 * (l2_value + l4_value)** 2) / 3
    i3 = (m3 * l3_value ** 2) / 3

    mi2 = -i2 * e2_value
    mi3 = -i3 * e3_value

    func1 = -g1*abs(((x_a_value + b_x_rounded)/2) - x_k_value)-f1_x*abs(y_k_value - ((y_a_value + b_y_rounded)/2))-f1_y*abs(((x_a_value + b_x_rounded)/2) - x_k_value)
    func2 = F(fi) - f2_y
    func3 = -f2_x*abs(y_k_value - ((e_y_rounded + b_y_rounded)/2)) - (f2_y + g2) * abs(((e_x_rounded + b_x_rounded)/2) - x_k_value) - F(fi) * abs(x_k_value - e_x_rounded) + mi2
    func4 =  -abs(x_k_value - ((c_x_rounded + x_d_value)/2))*(f3_y + g3) - f3_x * abs(y_k_value - ((c_y_rounded - y_d_value)/2)) + mi3 if -40 / 180 * math.pi + 2 * math.pi<=fi<=2 * math.pi else \
        abs(x_k_value - ((c_x_rounded + x_d_value) / 2)) * (f3_y + g3) + f3_x * abs(y_k_value - ((c_y_rounded - y_d_value) / 2)) - mi3
    B = np.array([0., 0., 0., 0., F(fi), -F(fi) * abs(x_k_value - e_x_rounded), 0., 0., 0.], dtype=np.float64)
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

    B1 = np.array([f1_x-1. if -40 / 180 * math.pi + 2 * math.pi<=fi<=2 * math.pi else -f1_x, -f1_y, func1, -f2_x, func2, func3, -f3_x, -f3_y, func4], dtype=np.float64)
    A1 = np.array([[1., 0., 1., 0., 0., 0., 0., 0., 0.],
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
    solve = np.linalg.solve(A1, B1)
    all_solve2.append(solve)
    solve = np.linalg.solve(A, B)
    all_solve1.append(solve)
print(all_solve2[0])
all_solve2 = np.transpose(all_solve2)
all_solve1 = np.transpose(all_solve1)
R = ["Rx10", "Ry10", "Rx12", "Ry12", "Rx23", "Ry23", "Rx30", "Ry30", "M1"]
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    print(all_solve2[i])
    degrees = [math.degrees(i) for i in np.arange(-40 / 180 * math.pi + 2 * math.pi, 40 / 180 * math.pi + 2 * math.pi, 0.001)]
    plt.plot(degrees, all_solve2[i], 'g')
    plt.plot(degrees, all_solve1[i], 'b')
    plt.xlabel('fi')
    plt.ylabel(R[i])
    plt.grid(True)
plt.tight_layout()

plt.figure()
for i in range(9):
    plt.plot(np.arange(-40 / 180 * math.pi + 2 * math.pi, 40 / 180 * math.pi + 2 * math.pi, 0.001), all_solve2[i], label=R[i])
    plt.xlabel('fi')
    plt.grid(True)
plt.legend()
plt.show()
