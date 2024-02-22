import sys
import fileinput

if len(sys.argv) == 1:
    for line in list(fileinput.input())[-17:]:
        print(line.rstrip())
elif len(sys.argv) == 2:
    file = sys.argv[1]
    with open(file) as f:
        for line in f.readlines()[-10:]:
            print(line.rstrip())
else:
    for file in sys.argv[1:]:
        print(f"==> {file} <==")
        with open(file) as f:
            for line in f.readlines()[-10:]:
                print(line.rstrip())
        print()
