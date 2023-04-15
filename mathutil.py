import random
import typing

Number = typing.Union[int, float]
TwoDimArray = typing.List[typing.List[Number]]


class Tensor:

    def __init__(
            self, m: int, n: int, matrix: typing.Optional[TwoDimArray] = None
    ) -> None:
        """Tensor with M rows and N cols"""
        self.m = m
        self.n = n
        self.matrix = matrix or [[0] * self.n for _ in range(self.m)]

    @classmethod
    def all(cls, m: int, n: int, val: Number) -> 'Tensor':
        return Tensor(m, n, [[val] * n for _ in range(m)])

    @classmethod
    def identity(cls, m: int, n: int, ) -> 'Tensor':
        res: TwoDimArray = [[0] * n for _ in range(m)]
        assert m == n, 'not an nxn matrix'
        for i in range(m):
            res[i][i] = 1
        return Tensor(m, n, res)

    def __getitem__(self, idx: int) -> typing.List[Number]:
        return self.matrix[idx]

    def __add__(self, other: object) -> 'Tensor':
        """Matrix addition"""
        if not isinstance(other, Tensor):
            raise TypeError(other)

        assert self.m == other.m and self.n == other.n

        res = []
        for i in range(self.m):
            row = []
            for j in range(self.n):
                row.append(self[i][j] + other[i][j])
            res.append(row)
        return Tensor(self.m, self.n, res)

    def __sub__(self, other: object) -> 'Tensor':
        """Matrix subtraction"""
        if not isinstance(other, Tensor):
            raise TypeError(other)

        assert self.m == other.m and self.n == other.n

        res = []
        for i in range(self.m):
            row = []
            for j in range(self.n):
                row.append(self[i][j] - other[i][j])
            res.append(row)
        return Tensor(self.m, self.n, res)

    def __mul__(self, other: object) -> 'Tensor':
        """Scalar Multiplication"""
        res: TwoDimArray = [[0] * self.n for _ in range(self.m)]
        if isinstance(other, int) or isinstance(other, float):
            for i in range(self.m):
                for j in range(self.n):
                    res[i][j] = self[i][j] * other
            return Tensor(self.m, self.n, res)

        if isinstance(other, Tensor):
            assert self.n == other.m
            res = [[0] * other.m for _ in range(self.m)]
            for i in range(self.m):
                for j in range(other.n):
                    for k in range(other.m):
                        res[i][j] += self[i][k] * other[k][j]
            return Tensor(self.m, other.m, res)

        raise TypeError(other)

    def __truediv__(self, other: Number) -> 'Tensor':
        """Scalar Division"""
        return self * (1 / other)

    def __floordiv__(self, other: object) -> 'Tensor':
        """Scalar Division"""
        res: TwoDimArray = [[0] * self.n for _ in range(self.m)]
        if isinstance(other, int) or isinstance(other, float):
            for i in range(self.m):
                for j in range(self.n):
                    res[i][j] = self[i][j] // other
            return Tensor(self.m, self.n, res)

        raise TypeError(other)

    def hadamard_product(self, other: object) -> 'Tensor':
        if not isinstance(other, Tensor):
            raise TypeError(other)

        assert self.m == other.m and self.n == other.n

        res: TwoDimArray = [[0] * self.n for _ in range(self.m)]
        for i in range(self.m):
            for j in range(self.n):
                res[i][j] = self[i][j] * other[i][j]

        return Tensor(self.m, self.n, res)

    def sum(self) -> Number:
        res: Number = 0
        for i in range(self.m):
            for j in range(self.n):
                res += self[i][j]
        return res

    def rotate(self) -> None:
        self.matrix = list(zip(*self.matrix))  # type: ignore

    def randomize(self, lb: Number, ub: Number) -> None:
        """Randomly choose values for every element of the matrix in the interval [lb:ub]
        including lb and ub."""
        for i in range(self.m):
            for j in range(self.n):
                self[i][j] = random.randint(int(lb), int(ub))
