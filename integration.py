class NotEnoughValueInY(Exception):
    pass


def trapezoidal_rule(y: list[float], h: float) -> float:
    """Finds the intergration using Trapezoidal Rule

    Params
    ------
    y: list[float]
        list of values of y
    h: float
        value of h, i.e., size of intervals
    Returns
    -------
    float:
        result
    """

    if len(y) < 2:
        raise NotEnoughValueInY

    return h*(y[0] + y[-1] + 2 * sum(y[1:-1]))/2


def simpson_1_3_rule(y: list[float], h: float) -> float:
    """Finds the intergration using Simpson's 1/3 Rule

    Params
    ------
    y: list[float]
        list of values of y
    h: float
        value of h, i.e., size of intervals
    Returns
    -------
    float:
        result
    """

    n = len(y) - 1
    if n < 2 or n % 2 != 0:
        raise NotEnoughValueInY

    odd_sum = sum(y[1:-1:2])
    even_sum = sum(y[2:-1:2])

    return h*(y[0] + y[-1] + 4*odd_sum + 2*even_sum)/3


def simpson_3_8_rule(y: list[float], h: float) -> float:
    """Finds the intergration using Simpson's 3/8 Rule

    Params
    ------
    y: list[float]
        list of values of y
    h: float
        value of h, i.e., size of intervals
    Returns
    -------
    float:
        result
    """

    n = len(y) - 1
    if n < 3 or n % 3 != 0:
        raise NotEnoughValueInY

    mul_3_sum = sum(y[3:-1:3])
    not_mul_3_sum = sum(y[1:-1]) - mul_3_sum

    return 3*h*(y[0] + y[-1] + 3*not_mul_3_sum + 2*mul_3_sum)/8


def waddle_rule(y: list[float], h: float) -> float:
    """Finds the intergration using Waddle's Rule

    Params
    ------
    y: list[float]
        list of values of y
    h: float
        value of h, i.e., size of intervals
    Returns
    -------
    float:
        result
    """

    n = len(y) - 1
    if n < 6 or n % 6 != 0:
        raise NotEnoughValueInY

    result = 0

    for i in range(0, n, 6):
        result += y[i] + 5*y[i+1] + y[i+2] + 6 * \
            y[i+3] + y[i+4] + 5*y[i+5] + y[i+6]

    result = (3*h*result)/10

    return result
