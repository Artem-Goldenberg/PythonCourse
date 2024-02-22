import sys
import fileinput

def count(input):
    lines = words = chars = 0
    for line in input:
        lines += 1
        chars += len(line)
        words += len(line.split())
    return lines, words, chars


if len(sys.argv) == 1:
    lines, words, chars = count(list(fileinput.input()))
    print(f"\t{lines}\t{words}\t{chars}")
else:
    tlines = twords = tchars = 0
    for file in sys.argv[1:]:
        with open(file) as f:
            lines, words, chars = count(f.readlines())
        print(f"\t{lines}\t{words}\t{chars}\t{file}")
        tlines += lines
        twords += words
        tchars += chars
    if len(sys.argv) > 2:
        print(f"\t{tlines}\t{twords}\t{tchars}\ttotal")

