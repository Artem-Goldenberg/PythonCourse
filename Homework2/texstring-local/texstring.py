import numpy as np


def finalForm(latex: str) -> str:
    header = r"""\documentclass[20pt]{extarticle}
    \usepackage{graphicx} % Required for inserting images

    \setlength{\tabcolsep}{20pt}
    \setlength{\arrayrulewidth}{0.4mm}
    \renewcommand{\arraystretch}{1.8}

    \begin{document}
    """

    footer = r"""
    \end{document}
    """

    return header + latex + footer


def table(table: np.ndarray) -> str:
    rows, cols = table.shape

    line = "\hline\n"
    newline = " \\\\\n"
    rowSep = newline + line

    formatString = "|c" * cols + "|"

    content = rowSep.join(" & ".join(map(str, row)) for row in table).join(
        (line, rowSep)
    )

    contextStart = r"\begin{center}\begin{tabular}"
    contextEnd = r"\end{tabular}\end{center}"
    return "%s{ %s }\n%s%s" % (contextStart, formatString, content, contextEnd)


def image(path: str) -> str:
    return r"""\begin{figure}[h]
    \includegraphics[width=16cm]{%s}
\end{figure}""" % path

