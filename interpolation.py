from abc import ABC, abstractmethod
from enum import Flag
from typing import Union
from IPython.display import HTML, display
from math import factorial


class Interpolation(ABC):
    """Abstract representation of Interpolation Methods"""

    def __init__(self, data: list[tuple[float, float]], precission: int = 5) -> None:
        """
        Params
        ------
        data : list[tuple[float, float]]
            data to be used for interpolation
        precission : int, optional
            precission to use while displaying the computation table (default is 5)
        """
        self.data = data
        self.data.sort()
        self.precission = precission
        self.h = data[1][0] - data[0][0]
        self.table_heading = ['x', 'y', '△y'] + \
            [f'△<sup>{i}</sup>y' for i in range(2, len(data))]
        self.computeDifferenceTable()

    @abstractmethod
    def getDiffYList(self, x) -> list[float]:
        """Returns the list containing the differences of y that will be used for computation

        Params
        ------
        x : float
            value of x for which difference list is to returned 
            (optional if difference list doesn't depened on value of x)

        Returns
        -------
        list[float]
            difference list
        """

    @abstractmethod
    def getNextPTerm(self, p: float, i: int) -> float:
        """Returns value that will be multiplied to pterm to update pterm

        Params
        ------
        p: float
            value of p
        i: int
            index of next term

        Returns
        -------
        float
            term to be multiplied
        """

    @abstractmethod
    def getX0Index(self, x) -> int:
        """Returns the index of x0

        Params
        ------
        x: float
            value of x for which x0 is to be computed (optional if x0 is not dependent on x)

        Returns
        -------
        int
            index of x0
        """

    def getP(self, x: float) -> float:
        """Returns the value of p for given x

        Params
        ------
        x: float
            value of x

        Returns
        -------
        float
            value of p
        """

        return (x - self.data[self.getX0Index(x)][0]) / self.h

    def solve(self, x: float) -> float:
        """To find the interpolation for given x

        Params
        ------
        x: float
            value of x

        Returns
        -------
        float
            y, i.e., interpolation of x
        """

        y, pterm = 0, 1
        p = self.getP(x)
        for i, dy in enumerate(self.getDiffYList(x)):
            y += (dy * pterm)/factorial(i)
            pterm *= self.getNextPTerm(p, i)
        return y

    def computeDifferenceTable(self):
        """Function to compute difference table."""
        self.table = []
        for x, _ in self.data:
            self.table.append([x])

        dy = [y for _, y in self.data]

        while len(dy) >= 1:
            for i, y in enumerate(dy):
                self.table[i].append(y)
            dy = [b - a for a, b in zip(dy, dy[1:])]

        self.__display__()

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


class NewtonForward(Interpolation):
    """Represent Newton's Forward Interpolation Method."""

    def getX0Index(self, x=None) -> int:
        return 0

    def getDiffYList(self, x=None) -> list[float]:
        return self.table[0][1:]

    def getNextPTerm(self, p: float, i: int) -> float:
        return p - i


class NewtonBackward(Interpolation):
    """Represent Newton's Backward Interpolation Method."""

    def getX0Index(self, x=None) -> int:
        if getattr(self, 'x0_ind', None) is None:
            self.x0_ind = len(self.data) - 1
        return self.x0_ind

    def getDiffYList(self, x=None) -> list[float]:
        return [row[-1] for row in self.table[::-1]]

    def getNextPTerm(self, p: float, i: int) -> float:
        return p + i


class CenterInterpolation(Interpolation, ABC):
    """Represent Center Interpolation Method."""

    def getX0Index(self, x: float) -> int:
        if getattr(self, 'x0_ind', None) is None or self.x0_ind[0] != x:
            x_dif = [abs(n-x) for n, _ in self.data]
            self.x0_ind = (x, x_dif.index(min(x_dif)))
        return self.x0_ind[1]


class GaussForward(CenterInterpolation):
    """Represent Gauss's Forward Interpolation Method."""

    def getDiffYList(self, x: float) -> list[float]:
        y_list = []
        for j in range(len(self.data)):
            try:
                y_list.append(self.table[self.getX0Index(x) - (j//2)][j+1])
            except IndexError:
                break
        return y_list

    def getNextPTerm(self, p: float, i: int) -> float:
        if i % 2 == 0:
            return p + (i//2)
        else:
            return p - (i//2 + 1)


class GaussBackward(CenterInterpolation):
    """Represent Gauss's Backward Interpolation Method."""

    def getDiffYList(self, x: float) -> list[float]:
        y_list = []
        for j in range(len(self.data)):
            try:
                y_list.append(self.table[self.getX0Index(x) - ((j+1)//2)][j+1])
            except IndexError:
                break
        return y_list

    def getNextPTerm(self, p: float, i: int) -> float:
        if i % 2 == 0:
            return p - (i//2)
        else:
            return p + (i//2 + 1)


class Stirling(CenterInterpolation):
    """Represent Stirling Interpolation Method."""

    def getDiffYList(self, x: float) -> list[float]:
        y_list = []
        for j in range(len(self.data)):
            try:
                y_list.append(
                    (self.table[self.getX0Index(x) - ((j+1)//2)][j+1]
                     + self.table[self.getX0Index(x) - (j//2)][j+1])/2
                )
            except IndexError:
                break
        return y_list

    def getNextPTerm(self, p: float, i: int) -> float:
        if i % 2 == 0:
            return (p*p - ((i//2)**2))/p
        else:
            return p


class LagrangeInterpolation:
    """Represents Lagrange's Interpolation method."""

    def __init__(self, data: list[tuple[float, float]], precission: int = 5) -> None:
        """
        Params
        ------
        data : list[tuple[float, float]]
            data to be used for interpolation
        precission : int, optional
            precission to use while displaying the computation table (default is 5)
        """

        self.data = data
        self.precission = precission
        self.table_heading = ['L(x)', 'y']

    def solve(self, x: float) -> float:
        """To find the interpolation for given x

        Params
        ------
        x: float
            value of x

        Returns
        -------
        float
            y, i.e., interpolation of x
        """

        self.table = []
        for xi, yi in self.data:
            Li = 1
            for xj, _ in self.data:
                if xi != xj:
                    Li *= (x - xj)/(xi - xj)
            self.table.append([Li, yi])

        self.__display__()
        return sum([Li*yi for Li, yi in self.table])

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


class NewtonDividedDifference:
    """Represents Newton's Divided Difference Interpolation method."""

    def __init__(self, data: list[tuple[float, float]], precission: int = 5) -> None:
        """
        Params
        ------
        data : list[tuple[float, float]]
            data to be used for interpolation
        precission : int, optional
            precission to use while displaying the computation table (default is 5)
        """

        self.data = data
        self.precission = precission
        self.table_heading = ['x', 'y', '◭y'] + \
            [f'◭<sup>{i}</sup>y' for i in range(2, len(data))]
        self.computeDividedDifferenceTable()

    def computeDividedDifferenceTable(self):
        """Function to compute difference table."""
        self.table = []
        for x, _ in self.data:
            self.table.append([x])

        dy = [y for _, y in self.data]

        j = 1
        while len(dy) >= 1:
            for i, y in enumerate(dy):
                self.table[i].append(y)

            dy = [(b - a)/(self.data[i+j][0] - self.data[i][0])
                  for i, (a, b) in enumerate(zip(dy, dy[1:]))]
            j += 1

        self.__display__()

    def solve(self, x: float) -> float:
        """To find the interpolation for given x

        Params
        ------
        x: float
            value of x

        Returns
        -------
        float
            y, i.e., interpolation of x
        """
        y, xterm = 0, 1
        for i, dy in enumerate(self.table[0][1:]):
            y += (dy * xterm)
            xterm *= (x - self.data[i][0])
        return y

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


if __name__ == "__main__":
    data = [(25, 4), (26, 3.846), (27, 3.704),
            (28, 3.571), (29, 3.448), (30, 3.333)]

    print(NewtonBackward(data).solve(27.5))
    print(NewtonForward(data).solve(27.5))
    print(Stirling(data).solve(27.5))
    print(GaussForward(data).solve(27.5))
    print(GaussBackward(data).solve(27.5))

    data = [(15, 24), (18, 37), (22, 25)]
    print(LagrangeInterpolation(data).solve(16))
    print(NewtonDividedDifference(data).solve(16))
