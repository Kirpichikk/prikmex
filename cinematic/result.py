import sympy as sp

x = sp.symbols('x ')

def result(x_value, x_evaluated, v_x, a_x, y_evaluated, v_y, a_y):
    x_rounded = x_evaluated.subs(x, x_value).evalf(2)
    y_rounded = y_evaluated.subs(x, x_value).evalf(2)
    v_x_rounded = v_x.subs(x, x_value).evalf(2)
    v_y_rounded = v_y.subs(x, x_value).evalf(2)
    a_x_rounded = a_x.subs(x, x_value).evalf(2)
    a_y_rounded = a_y.subs(x, x_value).evalf(2)

    return x_rounded, y_rounded, v_x_rounded, v_y_rounded, a_x_rounded, a_y_rounded