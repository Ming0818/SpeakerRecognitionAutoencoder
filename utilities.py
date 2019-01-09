import os
import random


def read_random_line(filename):
    file_size = os.stat(filename)[6] - 120000
    fd = open(filename, 'r+')
    line = ''
    for _ in range(10):  # Try 10 times
        pos1, pos2 = random.randint(0, file_size), random.randint(0, file_size)
        fd.seek(pos1)
        fd.readline()  # Read and ignore
        line = fd.readline()
        if line != '':
            fd.close()
            break
    if line != '' :
        return line
