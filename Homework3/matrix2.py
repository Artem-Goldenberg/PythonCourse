from functools import cache
import numpy as np

class MatrixAccessible:
    """Inserts array like 2D _data field"""

    def __init__(self, nparray: np.ndarray = None):
        if nparray is not None:
            assert nparray.ndim == 2
            self._data: list[list] = nparray.tolist()

    def __getitem__(self, i: tuple[int, int]):
        return self._data[i[0]][i[1]]

    def __setitem__(self, i: tuple[int, int], value):
        self._data[i[0]][i[1]] = value

    @property
    def rows(self):
        return len(self._data)

    @property
    def cols(self):
        return len(self._data[0])


class Arithmetic2D(MatrixAccessible, np.lib.mixins.NDArrayOperatorsMixin):
    def __array_ufunc__(self, ufunc: np.ufunc, method, *inputs, **kwargs):
        for x in inputs:
            if not isinstance(x, self.__class__):
                return NotImplemented
        a, b = inputs
        return self.__class__(ufunc(a._data, b._data))


class PrintableMatrix(MatrixAccessible):
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._data)})"

    def __str__(self):
        return "".join(str(l) + "\n" for l in self._data)


class IOWritableMatrix(PrintableMatrix):
    def write(self, path):
        with open(path, "w") as f:
            f.write(str(self))


class Matrix(IOWritableMatrix, Arithmetic2D):
    pass


if __name__ == "__main__":
    np.random.seed(0)
    a = Matrix(np.random.randint(0, 10, (10, 10)))
    b = Matrix(np.random.randint(0, 10, (10, 10)))
    dir = "artifacts/3.2"
    (a + b).write(f"{dir}/matrix+.txt")
    (a * b).write(f"{dir}/matrix*.txt")
    (a @ b).write(f"{dir}/matrix@.txt")
