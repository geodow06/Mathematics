from numpy import sin
import matplotlib.pyplot as plt

__all__ = ['rungeKutta4', 'euler', 'midpoint', 'pec']

def rungeKutta4(dydx, y0, x0, x, steps):
    """Solves ODE by Runge-Kutta using 4 increments

    given a function dydx(y(x),x) and initial conditions y0 = y(x0)

    returns an approximation for y(x) at the provided value of x
    """
    h = x / steps
    xn = x0
    yn = y0
    for n in range(0, steps):
        k1 = h * dydx(xn, yn)
        k2 = h * dydx(xn + 0.5 * h, yn + 0.5 * k1)
        k3 = h * dydx(xn + 0.5 * h, yn + 0.5 * k2)
        k4 = h * dydx(xn + h, yn + k3)
        yn = yn + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xn = xn + h
    return yn

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
