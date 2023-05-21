
import os
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(output_file, 'w') as fout:
    lines = open(input_file, 'rb').readlines()
    print(len(lines))
    for line in lines:
        line = line.strip().replace(b"\n", b"")
        fout.write('{}\n'.format(line))
