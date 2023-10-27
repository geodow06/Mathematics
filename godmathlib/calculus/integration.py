from numpy import sin
import matplotlib.pyplot as plt

__all__ = ['rk4', 'euler', 'midpoint', 'pec']


def rk4(dydx, y0, x0, x, steps):
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


def euler(dydx, y0, x0, x, steps):
    """Solves ODE by Euler method

    given a function dydx(y(x),x) and initial conditions y0 = y(x0)

    returns an approximation for y(x) at the provided value of x
    """
    h = x / steps
    yn = y0
    for n in range(steps):
        tn = x0 + n * h
        yn = yn + h * dydx(tn, yn)
    return yn


def midpoint(dydx, y0, x0, x, steps):
    """Solves ODE by Midpoint method

    given a function dydx(y(x),x) and initial conditions y0 = y(x0)

    returns an approximation for y(x) at the provided value of x
    """
    h = x / steps
    yn = y0
    for n in range(steps):
        t_n = x0 + n * h
        half_step = h / 2
        yn = yn + h * dydx(t_n + half_step, yn + half_step * dydx(t_n, yn))
    return yn


def pec(f):
    return True
