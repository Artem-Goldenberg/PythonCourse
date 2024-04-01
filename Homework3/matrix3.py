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
        result = self.__class__(ufunc(a._data, b._data))
        if ufunc.__name__ == "matmul" and hasattr(self, "cache"):
            self.cache[hash(a), hash(b)] = result
        return result


class PrintableMatrix(MatrixAccessible):
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._data)})"

    def __str__(self):
        return "".join(str(l) + "\n" for l in self._data)


class IOWritableMatrix(PrintableMatrix):
    def write(self, path):
        with open(path, "w") as f:
            f.write(str(self))


class HashableMatrix(MatrixAccessible):
    def __hash__(self):
        """Just sum everything up"""
        return sum(self[i, j] for i in range(self.rows) for j in range(self.cols))


class Matrix(IOWritableMatrix, Arithmetic2D, HashableMatrix):
    cache: dict[(int, int), 'Matrix'] = {}

    def fastMatMul(self, other):
        h1 = hash(self)
        h2 = hash(other)
        if (h1, h2) in Matrix.cache:
            return Matrix.cache[h1, h2]
        return self @ other


if __name__ == "__main__":
    a = Matrix(np.array([[1, 1], [-1, -1]]))
    b = Matrix(np.array([[2, 1], [1, 2]]))
    c = Matrix(np.array([[1, -1], [1, -1]]))
    d = b
    dir = "artifacts/3.3"

    a.write(f"{dir}/A.txt")
    b.write(f"{dir}/B.txt")
    c.write(f"{dir}/C.txt")
    d.write(f"{dir}/D.txt")
    (a.fastMatMul(b)).write(f"{dir}/AB.txt")
    (c @ d).write(f"{dir}/CD.txt")

    with open(f"{dir}/hash.txt", "w") as f:
        f.write(f"{hash(a @ b)=}\n")
        f.write(f"{hash(c @ d)=}\n")
