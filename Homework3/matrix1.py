import numpy as np


class Matrix:
    def __init__(self, nparray: np.ndarray = None):
        if nparray is not None:
            assert nparray.ndim == 2
            self._data: list[list] = nparray.tolist()

    def __getitem__(self, i: tuple[int, int]):
        return self._data[i[0]][i[1]]

    def __setitem__(self, i: tuple[int, int], value):
        self._data[i[0]][i[1]] = value

    def __str__(self):
        result = f"rows: {self.rows}, cols: {self.cols}\n"
        for l in self._data:
            result += str(l) + "\n"
        return result

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Columns or Rows do not match")
        a = Matrix()
        a._data = [
            [a + b for a, b in zip(l1, l2)] for l1, l2 in zip(self._data, other._data)
        ]
        return a

    def __mul__(self, other: "Matrix") -> "Matrix":
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Columns or Rows do not match")
        a = Matrix()
        a._data = [
            [a * b for a, b in zip(l1, l2)] for l1, l2 in zip(self._data, other._data)
        ]
        return a

    def __matmul__(self, other: "Matrix") -> "Matrix":
        if self.cols != other.rows:
            raise ValueError("Columns do not match rows")
        a = Matrix()
        a._data = [[0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                a[i, j] = sum(self[i, k] * other[k, j] for k in range(self.cols))
        return a
    
    @property
    def rows(self):
        return len(self._data)

    @property
    def cols(self):
        return len(self._data[0])


if __name__ == "__main__":
    np.random.seed(0)
    a = Matrix(np.random.randint(0, 10, (10, 10)))
    b = Matrix(np.random.randint(0, 10, (10, 10)))
    dir = "artifacts/3.1"
    with open(f"{dir}/matrix+.txt", "w") as pf, \
         open( f"{dir}/matrix*.txt", "w") as mf, \
         open(f"{dir}/matrix@.txt", "w") as mmf:
        pf.write(str(a + b))
        mf.write(str(a * b))
        mmf.write(str(a @ b))
