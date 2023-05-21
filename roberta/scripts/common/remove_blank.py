import os
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r', encoding='utf-8') as fin:
    fout = open('{}'.format(output_file), 'w', encoding='utf-8')
    for line in fin:
        history = None
        line_list = []
        for word in line.strip().split():
            if  history is not None and word == history:
                continue
            line_list.append(word)
            history = word
        fout.write('{}\n'.format(" ".join(line_list)))
