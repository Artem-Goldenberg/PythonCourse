import sys
import fileinput

for i, line in enumerate(fileinput.input(sys.argv[1:])):
    print(f"    {i + 1}  {line}", end="")
