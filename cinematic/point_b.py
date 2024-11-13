import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from result import result

# Определяем символы
x, l1, x_a, y_a = sp.symbols('x l1 x_a y_a')

def main():
    # Параметризованные функции
    b_x = "x_a + l1 * sin(x)"
    b_y = "y_a + l1 * cos(x)"
    b_x_func = sp.sympify(b_x)
    b_y_func = sp.sympify(b_y)

    l1_value = 1
    x_a_value = 2
    y_a_value = 3

    angles = np.linspace(0, 360, 361)  # Углы от 0 до 360 градусов  # Преобразуем в радианы

    positions_x = []
    positions_y = []
    velocities = []
    accelerations = []

    for angle in angles:
        x_value = np.radians(angle)

        b_x_evaluated = b_x_func.subs({l1: l1_value, x_a: x_a_value})
        b_y_evaluated = b_y_func.subs({l1: l1_value, y_a: y_a_value})

        v_b_x = sp.diff(b_x_evaluated, x)
        v_b_y = sp.diff(b_y_evaluated, x)
        a_b_x = sp.diff(v_b_x, x)
        a_b_y = sp.diff(v_b_y, x)


        pos = result(x_value, b_x_evaluated, v_b_x, a_b_x, b_y_evaluated, v_b_y, a_b_y)
        positions_x.append(pos[0])
        positions_y.append(pos[1])# x
        velocities.append(sp.sqrt(pos[2]**2 + pos[3]**2))  # V
        accelerations.append(sp.sqrt(pos[4]**2 + pos[5]**2))  # a

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
    plt.plot(range(10), velocities[:10], label='Скорость (V)', color='orange')
    plt.title('График скорости')
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Скорость (V)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(range(10), accelerations[:10], label='Ускорение (a)', color='red')
    plt.title('График ускорения')
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Ускорение (a)')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()