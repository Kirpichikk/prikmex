import sympy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt

# Определяем символы
x, l1, l2, l3, x_d, y_d, x_a, y_a, a, b, c, ad, db, fi_1, ac = sp.symbols('x l1 l2 l3 x_d y_d x_a y_a a b c ad db fi_1 ac')

def result(x_value, ang1, ang2, v2, a2, ang3, v3, a3):
    ang1_rounded = ang1.subs(x, x_value).evalf(2)
    ang2_rounded = ang2.subs(x, x_value).evalf(2)
    ang3_rounded = ang3.subs(x, x_value).evalf(2)
    v2_rounded = v2.subs(x, x_value).evalf(2)
    v3_rounded = v3.subs(x, x_value).evalf(2)
    a2_rounded = a2.subs(x, x_value).evalf(3)
    a3_rounded = a3.subs(x, x_value).evalf(3)

    return ang1_rounded, ang2_rounded, ang3_rounded, v2_rounded, v3_rounded, a2_rounded, a3_rounded

def main():
    T = 5/3
    w1 = (2 * m.pi) / T
    e1 = 0

    l1_value = 1
    l2_value = 2
    l3_value = 2

    x_d_value = 1
    y_d_value = 1

    x_a_value = 2
    y_a_value = 3

    ad_value = np.sqrt((x_d_value - x_a_value) ** 2 + (y_d_value - y_a_value) ** 2)

    c_valu = "pi / 2 - atan((abs(x_d - x_a)) / abs(y_d - y_a))"
    c_value_func = sp.simplify(c_valu)
    c_value = c_value_func.subs({x_d: x_d_value, x_a: x_a_value, y_d: y_d_value, y_a: y_a_value})
    c_value_degrees = m.degrees(c_value.evalf())
    c_value = c_value.evalf()
    angles = [m.degrees(i) for i in np.arange(-40 / 180 * m.pi + 2 * m.pi, 40 / 180 * m.pi + 2 * m.pi, 0.001)]  # Углы от 0 до 360 градусов

    i = 1.83
    j = 0

    angs = []
    velocities_of_link1 = []
    accelerations_of_link2 = []
    accelerations_of_link3 = []

    for angle in np.arange(-40 / 180 * m.pi + 2 * m.pi, 40 / 180 * m.pi + 2 * m.pi, 0.001):
        x_value = angle

        b_x = "x_a + l1 * cos(x)"
        b_y = "y_a - l1 * sin(x)"
        b_x_func = sp.sympify(b_x)
        b_y_func = sp.sympify(b_y)
        b_x_evaluated = b_x_func.subs({l1: l1_value, x_a: x_a_value})
        b_y_evaluated = b_y_func.subs({l1: l1_value, y_a: y_a_value})

        db_value = (b_x_evaluated - 1) ** 2 + (b_y_evaluated - 1) ** 2

        c_x = "x_d + l3 * cos(b + c - a)" if ((angle > (360 - c_value_degrees)) or (angle < (180 - c_value_degrees))) else "x_d + l3 * cos(b + c + a)"
        c_y = "y_d + l3 * sin(b + c - a)" if ((angle > (360 - c_value_degrees)) or (angle < (180 - c_value_degrees))) else "y_d + l3 * sin(b + c + a)"
        c_x_func = sp.sympify(c_x)
        c_y_func = sp.sympify(c_y)

        a_value = "acos((db - l1**2 + ad**2) / (2 * ad * db**0.5))"
        b_value = "acos((db - l2**2 + l3**2) / (2 * l3 * db**0.5))"
        a_value_func = sp.sympify(a_value)
        b_value_func = sp.sympify(b_value)
        b_value_evaluated = b_value_func.subs({db: db_value, l2: l2_value, l3: l3_value, ad: ad_value})
        a_value_evaluated = a_value_func.subs({db: db_value, l1: l1_value, ad: ad_value})

        c_x_evaluated = c_x_func.subs({l3: l3_value, x_d: x_d_value, b: b_value_evaluated, c: c_value, a: a_value_evaluated})
        c_y_evaluated = c_y_func.subs({l3: l3_value, y_d: y_d_value, b: b_value_evaluated, c: c_value, a: a_value_evaluated})

        ac_value = (c_x_evaluated - x_a_value) ** 2 + (c_y_evaluated - y_a_value) ** 2

        # углы, определяющие поворот каждого из 3х звеньев
        fi_1_value = x
        fi_2 = "pi - x + acos((l1**2 + l2**2 - ac) / (2 * l1 * l2))"
        fi_2_func = sp.sympify(fi_2)
        fi_2_value = fi_2_func.subs({l1: l1_value, l2: l2_value, ac: ac_value})
        fi_3_value = b_value_evaluated + c_value - a_value_evaluated if ((angle > (360 - c_value_degrees)) or (angle < (180 - c_value_degrees))) else b_value_evaluated + c_value - a_value_evaluated

        wq2 = sp.diff(fi_2_value, fi_1_value)
        w2 = wq2 * w1
        eq2 = sp.diff(wq2, fi_1_value)
        e2 = eq2 * w1 ** 2 + wq2 * e1
        wq3 = sp.diff(fi_3_value, fi_1_value)
        w3 = wq3 * w1
        eq3 = sp.diff(wq3, fi_1_value)
        e3 = eq3 * w1 ** 2 + wq3 * e1

        pos = result(x_value, fi_1_value, fi_2_value, w2, e2, fi_3_value, w3, e3)

        angs.append(pos[0])
        angs.append(pos[1])
        angs.append(pos[2])

        velocities_of_link1.append(w1)
        if not(angle == 0 or angle == 360):
            accelerations_of_link2.append(pos[5])
        else:
            accelerations_of_link2.append(0)
        accelerations_of_link3.append(pos[6])

    plt.figure()
    plt.plot(angles, velocities_of_link1, label='Скорость (w1)', color='orange')
    plt.title('График скорости')
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Скорость (w1)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(angles, accelerations_of_link2, label='Ускорение (e2)', color='red')
    # plt.scatter(angles, accelerations, color='red')
    plt.title('График ускорения')
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Ускорение (e2)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(angles, accelerations_of_link3, label='Ускорение (e3)', color='red')
    # plt.scatter(angles, accelerations, color='red')
    plt.title('График ускорения')
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Ускорение (e3)')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()