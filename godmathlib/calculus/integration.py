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
    yn = y0
    for n in range(steps):
        tn = t0 + n * h
        yn = yn + h * dydt(tn, yn)
    return yn

def midpoint(dydt, y0, t0, t, steps):
    """Solves ODE by Midpoint method

    given a function dydt(y(t),t) and initial conditions y0 = y(t0)
    """
    h = t / steps
    yn = y0
    for n in range(steps):
        t_n = t0 + n * h
        half_step = h / 2
        yn = yn + h * dydt(t_n + half_step, yn + half_step * dydt(t_n, yn))
    return yn

def pec(f):
    return True
