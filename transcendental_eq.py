#!/usr/bin/env python3
from typing import Callable
from abc import ABC, abstractmethod
from IPython.display import HTML, display


class RootDoesNotExit(Exception):
    """Exception to be raised when root does not exist for the given set of values"""


class TranscendentalEq(ABC):
    """Abstract Base class to represent Transcendental Equation Solver"""

    def __init__(self, f: Callable[[float], float], precission: int = 5, table_heading=None) -> None:
        """
        Params
        ------
        f : Callable
            function that represent the transcendental equation
        precission : int, optional
            precission to use while displaying the computation table (default is 5)
        """

        self.f = f
        self.precission = precission
        self.table_heading = table_heading or ["a", "b", "c", "f(c)"]

    @abstractmethod
    def getNextX(self, initial_values: list[float]):
        """Function that computes the next value of x"""

    def deco(self) -> Callable[[list[float]], tuple[float, float]]:
        """Function to decorate the transcendental equation to calcuate
        next value of x and display result of each computation."""

        def inner(initial_values: list[float]) -> tuple[float, float]:
            x = self.getNextX(initial_values)  # get next value of x
            y = self.f(x)       # get value of f(x)

            # add elements to table
            self.table.append([*initial_values, x, y])
            return x, y

        return inner

    def solve(self, neg: float, pos: float, ep: float, iter: int = 100) -> float:
        """Function to find the root of equation.

        Params
        ------
        neg: float
            value of x for which f(x) is negative
        pos: float
            value of x for which f(x) is positive
        ep: float
            value of epsilon
        iter: int, optional
            maxinum number of iterations to perform (default is 100)

        Returns
        -------
        float
            root of the equation
        """

        self.table = []     # initialize the table
        neg_val, pos_val = self.f(neg), self.f(pos)
        if abs(neg_val) <= ep:
            return neg
        elif abs(pos_val) <= ep:
            return pos
        elif neg_val > 0 and pos_val < 0:
            neg, pos = pos, neg
            neg_val, pos_val = pos_val, neg_val
        elif neg_val > 0 or pos_val < 0:
            raise RootDoesNotExit

        f = self.deco()     # get decorated f()

        for _ in range(iter):
            x, val = f([neg, pos])
            if abs(val) <= ep:
                break
            elif val < 0:
                neg_val = val
                neg = x
            elif val > 0:
                pos_val = val
                pos = x
            else:
                break
        self.__display__()
        return neg if abs(neg_val) < abs(pos_val) else pos

    def __display__(self):
        display(
            HTML(
                f'''<table>
                    <tr>{"".join(f'<th>{h}</th>' for h in self.table_heading)}</tr>
                    {"".join(
                        f'<tr>{"".join(f"<td>{val:.{self.precission}f}</td>" for val in row)}</tr>' for row in self.table
                    )}
                </table>'''
            ))


class BisectionMethod(TranscendentalEq):
    """Solver that solves Transcendental Equation by Bisection method."""

    def getNextX(self, initial_values: list[float]) -> float:
        a, b = initial_values
        return (a + b)/2


class FalsePositionMethod(TranscendentalEq):
    """Solver that solves Transcendental Equation by False Position method
    or Regular Falsi method."""

    def getNextX(self, initial_values: list[float]) -> float:
        a, b = initial_values
        return (b * self.f(a) - a * self.f(b))/(self.f(a) - self.f(b))


class NewtonRaphsonMethod(TranscendentalEq):
    """Solver that solves Transcendental Equation by Newton Raphson method."""

    def __init__(self, f: Callable[[float], float], fd: Callable[[float], float], *args, **kwargs) -> None:
        """
        Params
        ------
        f : Callable
            function that represent the transcendental equation
        fd : Callable
            function that represent the differential of transcendental equation
        *args:
            Positional Arguments
        **kwargs:
            Keyword Arguments
        """

        super().__init__(f, table_heading=[
            "x<sub>i-0</sub>", "x<sub>i</sub>", "f(x<sub>i</sub>)"], *args, **kwargs)
        self.fd = fd

    def getNextX(self, initial_values: list[float]) -> float:
        a, = initial_values
        return a - (self.f(a) / self.fd(a))

    def solve(self, start: float, ep: float, iter: int = 100) -> float:
        """Function to find the root of equation.

        Params
        ------
        start: float
            initial value of x
        ep: float
            value of epsilon
        iter: int, optional
            maxinum number of iterations to perform (default is 100)

        Returns
        -------
        float
            root of the equation
        """
        self.table = []
        f = self.deco()

        for _ in range(iter):
            x, y = f([start])
            if abs(y) <= ep or abs(self.fd(x)) == 0:
                break
            else:
                start = x

        self.__display__()
        return x


class SecantMethod(TranscendentalEq):
    """Solver that solves Transcendental Equation by Secant method."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(table_heading=[
            "x<sub>i-0</sub>", "x<sub>i</sub>", "x<sub>i+1</sub>", "f(x<sub>i+1</sub>)"], *args, **kwargs)

    def getNextX(self, initial_values: list[float]) -> float:
        a, b = initial_values
        return (b * self.f(a) - a * self.f(b))/(self.f(a) - self.f(b))

    def solve(self, a: float, b: float, ep: float, iter: int = 100) -> float:
        """Function to find the root of equation.

        Params
        ------
        a: float
            first initial value of x
        b: float
            second initial value of x
        ep: float
            value of epsilon
        iter: int, optional
            maxinum number of iterations to perform (default is 100)

        Returns
        -------
        float
            root of the equation
        """
        self.table = []
        f = self.deco()

        for _ in range(iter):
            x, val = f([a, b])
            if abs(val) <= ep:
                break
            else:
                a, b = b, x

        self.__display__()
        return x


def iterative_method(f, start: float = 0, precission: int = 5, ep: float = 1e-8, iter: int = 100):
    x = f(start)
    table = [[start, x]]
    for _ in range(iter):
        y = f(x)
        table.append([x, y])
        if(abs(x-y) <= ep):
            break
        x = y
    display(
        HTML(
            f'''<table>
                    <tr>{"".join(f'<th>{h}</th>' for h in ["x", "f(x)"])}</tr>
                    {"".join(
                        f'<tr>{"".join(f"<td>{val:.{precission}f}</td>" for val in row)}</tr>' for row in table
                    )}
                </table>'''
        ))
    return x


def f(x: float) -> float:
    return x**3-2*x-5


def fd(x: float) -> float:
    return 3*x**2-2


if __name__ == "__main__":
    print(BisectionMethod(f).solve(2, 3, 1e-5))
    print(FalsePositionMethod(f).solve(2, 3, 1e-5))
    print(NewtonRaphsonMethod(f, fd).solve(3, 1e-5))
    print(SecantMethod(f).solve(3, 2.5, 1e-5))
