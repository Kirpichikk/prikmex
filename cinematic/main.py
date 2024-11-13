import sympy as sp
import matplotlib.pyplot as plt

# Определяем символы
x, l1, l2, l3, l4, a, b, c, ad, db, x_c, x_b, y_c, y_b, x_a, y_a, x_d, y_d = sp.symbols(
    'x l1 l2 l3 l4 a b c ad db x_c x_b y_c y_b x_a y_a x_d y_d')


# Функция для вычисления результатов
def result(x_evaluated, v_x, a_x, y_evaluated, v_y, a_y):
    x_rounded = x_evaluated.evalf(2)
    y_rounded = y_evaluated.evalf(2)
    v_x_rounded = v_x.evalf(2)
    v_y_rounded = v_y.evalf(2)
    a_x_rounded = a_x.evalf(2)
    a_y_rounded = a_y.evalf(2)

    V = sp.sqrt(v_x_rounded ** 2 + v_y_rounded ** 2)
    a = sp.sqrt(a_x_rounded ** 2 + a_y_rounded ** 2)

    return x_rounded, y_rounded, V, a


def compute_b_coordinates(x_value, l1_value, x_a_value, y_a_value):
    b_x = x_a_value + l1_value * sp.sin(x_value)
    b_y = y_a_value + l1_value * sp.cos(x_value)

    v_b_x = sp.diff(b_x, x)
    v_b_y = sp.diff(b_y, x)
    a_b_x = sp.diff(v_b_x, x)
    a_b_y = sp.diff(v_b_y, x)

    return b_x, b_y, v_b_x, v_b_y, a_b_x, a_b_y


def compute_c_coordinates(x_value, b_x_evaluated, b_y_evaluated, l2_value, l3_value, ad_value, x_d_value, y_d_value):
    db_value = (b_x_evaluated - 1) ** 2 + (b_y_evaluated - 1) ** 2
    b_value_evaluated = 180 - sp.acos((l2_value ** 2 - l3_value ** 2 - db_value) / (2 * l3_value * sp.sqrt(db_value)))
    a_value_evaluated = 180 - sp.acos((l1 ** 2 - ad_value ** 2 - db_value) / (2 * ad_value * sp.sqrt(db_value)))

    if (x_value > sp.rad(90) - sp.rad(63.4349488229)) and (x_value < sp.rad(270) - sp.rad(63.4349488229)):
        c_x = x_d_value + l3_value * sp.cos(b_value_evaluated + sp.rad(63.4349488229) - a_value_evaluated)
        c_y = y_d_value + l3_value * sp.sin(b_value_evaluated + sp.rad(63.4349488229) - a_value_evaluated)
    else:
        c_x = x_d_value + l3_value * sp.cos(b_value_evaluated + sp.rad(63.4349488229) + a_value_evaluated)
        c_y = y_d_value + l3_value * sp.sin(b_value_evaluated + sp.rad(63.4349488229) + a_value_evaluated)

    v_c_x = sp.diff(c_x, x)
    v_c_y = sp.diff(c_y, x)
    a_c_x = sp.diff(v_c_x, x)
    a_c_y = sp.diff(v_c_y, x)

    return c_x, c_y, v_c_x, v_c_y, a_c_x, a_c_y


def compute_e_coordinates(e_x_func, e_y_func, l2_value, l4_value, c_x_evaluated, b_x_evaluated, c_y_evaluated,
                          b_y_evaluated):
    e_x_evaluated = e_x_func.subs({l2: l2_value, l4: l4_value, x_c: c_x_evaluated, x_b: b_x_evaluated})
    e_y_evaluated = e_y_func.subs({l2: l2_value, l4: l4_value, y_c: c_y_evaluated, y_b: b_y_evaluated})

    v_e_x = sp.diff(e_x_evaluated, x)
    v_e_y = sp.diff(e_y_evaluated, x)
    a_e_x = sp.diff(v_e_x, x)
    a_e_y = sp.diff(v_e_y, x)

    return e_x_evaluated, e_y_evaluated, v_e_x, v_e_y, a_e_x, a_e_y


def main():
    # Константы
    l1_value, l2_value, l3_value, l4_value = 1, 2, 2, 3
    x_a_value, y_a_value, x_d_value, y_d_value = 2, 3, 1, 1
    ad_value = 2.2360679775

    # Параметризованные функции
    e_x_func = sp.sympify("x_c + ((l4 * (x_c - x_b)) / (l2))")
    e_y_func = sp.sympify("y_c + ((l4 * (y_c - y_b)) / (l2))")

    # Списки для хранения результатов
    b_x_plot, b_y_plot, c_x_plot, c_y_plot = [], [], [], []
    v_b_plot, v_c_plot, a_b_plot, a_c_plot = [], [], [], []
    e_x_plot, e_y_plot, v_e_plot, a_e_plot = [], [], [], []

    for i in range(0, 361):
        x_value = sp.rad(i)

        # Вычисление координат b
        b_x_evaluated, b_y_evaluated, v_b_x, v_b_y, a_b_x, a_b_y = compute_b_coordinates(x_value, l1_value, x_a_value, y_a_value)
        point_b_x, point_b_y, _, _ = result(b_x_evaluated, v_b_x, a_b_x, b_y_evaluated, v_b_y, a_b_y)
        b_x_plot.append(point_b_x)
        b_y_plot.append(point_b_y)
        v_b_plot.append(sp.sqrt(v_b_x.evalf(2)**2 + v_b_y.evalf(2)**2))
        a_b_plot.append(sp.sqrt(a_b_x.evalf(2)**2 + a_b_y.evalf(2)**2))

        # Вычисление координат c
        c_x_evaluated, c_y_evaluated, v_c_x, v_c_y, a_c_x, a_c_y = compute_c_coordinates(x_value, b_x_evaluated, b_y_evaluated, l2_value, l3_value, ad_value, x_d_value, y_d_value)
        point_c_x, point_c_y, _, _ = result(c_x_evaluated, v_c_x, a_c_x, c_y_evaluated, v_c_y, a_c_y)
        c_x_plot.append(point_c_x)
        c_y_plot.append(point_c_y)
        v_c_plot.append(sp.sqrt(v_c_x.evalf(2)**2 + v_c_y.evalf(2)**2))
        a_c_plot.append(sp.sqrt(a_c_x.evalf(2)**2 + a_c_y.evalf(2)**2))

        # Вычисление координат e
        e_x_evaluated, e_y_evaluated, v_e_x, v_e_y, a_e_x, a_e_y = compute_e_coordinates(e_x_func, e_y_func, l2_value, l4_value, c_x_evaluated, b_x_evaluated, c_y_evaluated, b_y_evaluated)
        point_e_x, point_e_y, _, _ = result(e_x_evaluated, v_e_x, a_e_x, e_y_evaluated, v_e_y, a_e_y)
        e_x_plot.append(point_e_x)
        e_y_plot.append(point_e_y)
        v_e_plot.append(sp.sqrt(v_e_x.evalf(2)**2 + v_e_y.evalf(2)**2))
        a_e_plot.append(sp.sqrt(a_e_x.evalf(2)**2 + a_e_y.evalf(2)**2))

    plt.figure()
    plt.plot(b_x_plot, b_y_plot, 'r', label='Звено b')
    plt.plot(c_x_plot, c_y_plot, 'b', label='Звено c')
    plt.plot(e_x_plot, e_y_plot, 'g', label='Точка e')
    plt.title('Графики звеньев b, c и точки e')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    # График скорости
    plt.figure()
    plt.plot(range(0, 361), v_b_plot, 'r', label='Скорость b')
    plt.plot(range(0, 361), v_c_plot, 'b', label='Скорость c')
    plt.plot(range(0, 361), v_e_plot, 'g', label='Скорость e')
    plt.title('Графики скорости от угла')
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Скорость')
    plt.legend()
    plt.grid()
    plt.show()

    # График ускорения
    plt.figure()
    plt.plot(range(0, 361), a_b_plot, 'r', label='Ускорение b')
    plt.plot(range(0, 361), a_c_plot, 'b', label='Ускорение c')
    plt.plot(range(0, 361), a_e_plot, 'g', label='Ускорение e')
    plt.title('Графики ускорения от угла')
    plt.xlabel('Угол (градусы)')
    plt.ylabel('Ускорение')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()