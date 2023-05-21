import os
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r', encoding='utf-8') as fin:
    fout = open('{}'.format(output_file), 'w', encoding='utf-8')
    for line in fin:
        line = line.strip().replace(".", ". ").replace("?", " ? ").replace("\"", " \" ").replace("!", " ! ").strip()
        line = " ".join(line.split())
        fout.write('{}\n'.format(line))
