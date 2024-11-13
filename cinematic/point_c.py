import sympy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt
from result import result

# Определяем символы
x, l1, l2, l3, x_d, y_d, x_a, y_a, a, b, c, ad, db= sp.symbols('x l1 l2 l3 x_d y_d x_a y_a a b c ad db')

def main():

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
    angles = np.linspace(0, 360, 361)  # Углы от 0 до 360 градусов  # Преобразуем в радианы

    positions_x = []
    positions_y = []
    velocities = []
    accelerations = []

    for angle in angles:
        x_value = np.radians(angle)

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

        v_c_x = sp.diff(c_x_evaluated, x)
        v_c_y = sp.diff(c_y_evaluated, x)
        a_c_x = sp.diff(v_c_x, x)
        a_c_y = sp.diff(v_c_y, x)

        pos = result(x_value, c_x_evaluated, v_c_x, a_c_x, c_y_evaluated, v_c_y, a_c_y)
        positions_x.append(pos[0])
        positions_y.append(pos[1])  #  # x
        velocities.append(sp.sqrt(pos[2]**2 + pos[3]**2)) # V
        accelerations.append(sp.sqrt(pos[4]**2 + pos[5]**2)) # a

    print(positions_x)
    print(positions_y)
    # Построение графиков
    plt.figure()
    plt.plot(positions_x, positions_y, label='Положение (x)')
    plt.title('График перемещения')
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Положение (x)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(angles, velocities, label='Скорость (V)', color='orange')
    plt.title('График скорости')
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Скорость (V)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(angles, accelerations, label='Ускорение (a)', color='red')
    plt.title('График ускорения')
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Ускорение (a)')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()