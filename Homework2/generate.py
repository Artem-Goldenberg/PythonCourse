import numpy as np
import texstring as tex

def doit():
    a = np.arange(56).reshape(7, 8)

    with open("artifacts/table.tex", "w") as f:
        f.write(tex.finalForm(tex.table(a) + tex.image("progression.png") + tex.image("graph.png")))


if __name__ == "__main__":
    doit()
