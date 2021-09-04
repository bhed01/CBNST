from math import factorial
from typing import Callable


def taylor_method(y: list[float], x0: float, x: float) -> float:
    """Finds the value of y(x) for some ODE using Taylor's Method.

    Params
    ------
    y: list[float]
        list of value of different order differenciation of y at x0
    x0: float
        value of x0
    x: float
        value at which value y is to be computed

    Returns
    -------
    float
        value of y(x)
    """
    h = x - x0
    return sum([(h**i*yi)/factorial(i) for i, yi in enumerate(y)])


def euler_method(f: Callable[[float, float], float], y0: float, x: list[float], h: float) -> float:
    """Finds the value of y(x) for some ODE using Euler's Method.

    Params
    ------
    f: Callable[[float, float], float]
        function representing dy/dx, i.e., f(x, y)
    y0: float
        value of y at x0
    x: list[float]
        list of values at which y has to be computed
    h: float
        interval size

    Returns
    -------
    float
        value of y(x)
    """

    yi = y0
    for xi in x[:-1]:
        yi = yi + h*f(xi, yi)

    return yi


def euler_modified_method(f: Callable[[float, float], float], y0: float, x: list[float], h: float, m: int = 1) -> float:
    """Finds the value of y(x) for some ODE using Euler's Modified Method.

    Params
    ------
    f: Callable[[float, float], float]
        function representing dy/dx, i.e., f(x, y)
    y0: float
        value of y at x0
    x: list[float]
        list of values at which y has to be computed
    h: float
        interval size
    m: int, optional
        number of modifications to be made

    Returns
    -------
    float
        value of y(x)
    """

    prev_yi, yi, yi_m = y0, y0, y0
    for xi, next_xi in zip(x[:-1], x[1:]):
        prev_yi = yi
        yi = yi + h*f(xi, yi)
        yi_m = yi
        for _ in range(m):
            yi_m = prev_yi + h*(f(xi, prev_yi) + f(next_xi, yi_m))/2

    return yi_m


def runge_kutta_method_o2(f: Callable[[float, float], float], y0: float, x: list[float], h: float, precission: int = 5) -> float:
    """Finds the value of y(x) for some ODE using Runge Kutta method (order 2).

    Params
    ------
    f: Callable[[float, float], float]
        function representing dy/dx, i.e., f(x, y)
    y0: float
        value of y at x0
    x: list[float]
        list of values at which y has to be computed
    h: float
        interval size

    Returns
    -------
    float
        value of y(x)
    """

    yi = y0
    for i, xi in enumerate(x[:-1], 1):
        k1 = h*f(xi, yi)
        k2 = h*f(xi+h, yi+k1)
        yi = yi + (k1 + k2)/2
        print(f"y{i}={yi:.{precission}f}\n\t{k1=:.{precission}f}\n\t{k2=:.{precission}f}")

    return yi


def runge_kutta_method_o4(f: Callable[[float, float], float], y0: float, x: list[float], h: float, precission: int = 5) -> float:
    """Finds the value of y(x) for some ODE using Runge Kutta method (order 4).

    Params
    ------
    f: Callable[[float, float], float]
        function representing dy/dx, i.e., f(x, y)
    y0: float
        value of y at x0
    x: list[float]
        list of values at which y has to be computed
    h: float
        interval size
    

    Returns
    -------
    float
        value of y(x)
    """

    yi = y0
    for i, xi in enumerate(x[:-1], 1):
        k1 = h*f(xi, yi)
        k2 = h*f(xi + h/2, yi + k1/2)
        k3 = h*f(xi + h/2, yi + k2/2)
        k4 = h*f(xi + h, yi + k3)
        yi = yi + (k1 + 2*k2 + 2*k3 + k4)/6
        print(f"y{i}={yi:.{precission}f}\n\t{k1=:.{precission}f}\n\t{k2=:.{precission}f}\n\t{k3=:.{precission}f}\n\t{k4=:.{precission}f}")

    return yi


def predictor_formula(y0: float, h: float, y1_: float, y2_: float, y3_: float) -> float:
    return y0 + 4*h*(2*y1_ - y2_ + 2*y3_)/3


def corrector_formula(y2: float, h: float, y2_: float, y3_: float, y4_: float) -> float:
    return y2 + h*(y2_ + 4*y3_ + y4_)/3


def milne_predictor_corrector_method(f: Callable[[float, float], float],
                                     y0: float,
                                     x: list[float],
                                     h: float,
                                     yl: list[float]) -> float:
    """Finds the value of y(x) for some ODE using Milne's Predictor Corrector Method.

    Params
    ------
    f: Callable[[float, float], float]
        function representing dy/dx, i.e., f(x, y)
    y0: float
        value of y at x0
    x: list[float]
        list of values at which y has to be computed
    h: float
        interval size
    yl: list[float]
        value of y1, y2 and y3 calculate by any other method

    Returns
    -------
    float
        value of y(x)
    """
    yl = yl[:3]
    yl_ = [f(xi, yi) for xi, yi in zip(x[1:4], yl)]

    yi = y0
    for i, xi in enumerate(x[4:]):
        yi = predictor_formula(y0, h, yl_[i], yl_[i+1], yl_[i+2])
        yi_ = f(xi, yi)

        yl.append(yi)
        yl_.append(yi_)

        yi = corrector_formula(yl[i+1], h, yl_[i+1], yl_[i+2], yl_[i+3])

    return yi
