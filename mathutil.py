import math
import typing


class Tensor:

    def __init__(
            self, m: int, n: int, matrix: typing.Optional[typing.List[typing.List[int]]] = None
    ) -> None:
        """Tensor with M rows and N cols"""
        self.m = m
        self.n = n
        self.matrix = matrix or [[0] * self.n for _ in range(self.m)]

    def __getitem__(self, idx: int) -> typing.List[int]:
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
        res = [[0] * self.n for _ in range(self.m)]
        if isinstance(other, int) or isinstance(other, float):
            for i in range(self.m):
                for j in range(self.n):
                    # round down because only ints are supported for now
                    res[i][j] = math.floor(self[i][j] * other)
            return Tensor(self.m, self.n, res)

        raise TypeError(other)

    def rotate(self) -> None:
        self.matrix = list(zip(*self.matrix))  # type: ignore
