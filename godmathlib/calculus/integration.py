from numpy import sin
import matplotlib.pyplot as plt

__all__ = ['rungeKutta', 'euler', 'midpoint', 'pec']


def rungeKutta(dydx, y0, x0, x, steps):
    # Count number of iterations using step size or
    h = x / steps
    # Iterate for number of iterations
    y = y0
    for i in range(1, steps + 1):
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * dydx(x0, y)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * dydx(x0 + h, y + k3)

        # Update next value of y
        y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Update next value of x
        x0 = x0 + h
    return y


def euler(dydt, y0, t0, t, steps):
    """Solves ODE by Euler method

    given a function dydt(y(t),t) and initial conditions y0 = y(t0)
    """
    h = t / steps
    t_values = [t0]
    y_values = [y0]
    for n in range(steps + 1):
        if n > 0:
            new_t = t_values[0] + n * h
            new_y = y_values[n - 1] + h * dydt(new_t, y_values[n - 1])
            t_values.append(new_t)
            y_values.append(new_y)
    plt.plot(t_values, y_values, label="Euler")
    y = y_values[-1]
    return y


def midpoint(dydt, y0, t0, t, steps):
    """Solves ODE by Midpoint method

    given a function dydt(y(t),t) and initial conditions y0 = y(t0)
    """
    t_values = [t0]
    y_values = [y0]
    h = t / steps
    for n in range(steps + 1):
        t_n = t_values[0] + n * h
        half_step = h / 2
        y_n_plus_one = y_values[n] + h * dydt(t_n + half_step, y_values[n] + half_step * dydt(t_n, y_values[n]))
        t_values.append(t_n + h)
        y_values.append(y_n_plus_one)
    plt.plot(t_values, y_values, label="Midpoint")
    y = y_values[-1]
    return y


def pec(f):
    return True
