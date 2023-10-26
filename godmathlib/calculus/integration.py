from numpy import sin
import matplotlib.pyplot as plt

__all__ = ['rungeKutta', 'euler', 'midpoint', 'pec']


def rungeKutta(dydx, x0, y0, x, h):
    # Count number of iterations using step size or
    # step height h
    n = (int)((x - x0) / h)
    # Iterate for number of iterations
    y = y0
    for i in range(1, n + 1):
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


def euler(dydt, y0, t0, steps, target_t):
    """"Solves ODE by Euler method

    given a function dydt(y(t),t) and initial conditions y0 = y(t0)
    """
    h = target_t / steps
    t = [t0]
    y = [y0]
    for n in range(steps + 1):
        if n > 0:
            new_t = t[0] + n * h
            new_y = y[n - 1] + h * dydt(new_t, y[n - 1])
            t.append(new_t)
            y.append(new_y)
    plt.plot(t, y, label="Euler")
    target_y = y[-1]
    return target_y


def midpoint(dydt, y0, t0, steps, target_t):
    """"Solves ODE by Midpoint method

    given a function dydt(y(t),t) and initial conditions y0 = y(t0)
    """
    t = [t0]
    y = [y0]
    h = target_t / steps
    for n in range(steps + 1):
        t_n = t[0] + n * h
        half_step = h / 2
        y_n_plus_one = y[n] + h * dydt(t_n + half_step, y[n] + half_step * dydt(t_n, y[n]))
        t.append(t_n + h)
        y.append(y_n_plus_one)
    plt.plot(t, y, label="Midpoint")
    target_y = y[-1]
    return target_y


def pec(f):
    return True
