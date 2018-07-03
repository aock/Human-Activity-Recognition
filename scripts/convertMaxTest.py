from src.saveFunctions import *
import sys
import glob
import ntpath
import re
import numpy as np
ntpath.basename("a/b/c")


IN_FOLDER = 'import/NRM/man_gen'
OUT_FOLDER = 'export/NRM/man_gen'

OUT_TEMPLATE = './' + OUT_FOLDER + '/NRM_%d.tdat'

# model for prediction

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def gen_filename(i):
    return OUT_TEMPLATE % i

def join(l, sep):
    out_str = ''
    for i, el in enumerate(l):
        out_str += '%s%s' % (el, sep)
    return out_str[:-len(sep)]

def get_2d_list_slice(matrix, start_row, end_row, start_col, end_col):
    return [row[start_col:end_col] for row in matrix[start_row:end_row]]


def dataToString(data):
    filestr = ''
    for i,row in enumerate(data):
        filestr += join(row, ',') + '\n'
    return filestr


if __name__ == "__main__":
    files = [filename for filename in glob.iglob(IN_FOLDER + '/**/*.csv', recursive=True)]
    print(files)

    pattern = r'(^[a-zA-Z]+)_.*\.([a-zA-Z]+$)'
    prog = re.compile(pattern)

    for i,filepath in enumerate(files):

        # read data
        in_data = []
        with open(filepath, 'r') as in_file:
            for line in in_file:
                try:
                    ll = line.split(',')
                    entry = ['NoStairs', float(ll[1]), float(ll[2]), float(ll[3]) ]
                    in_data.append(entry)
                except:
                    pass

        # convert data
        out_data = in_data
        out_data_str = dataToString(out_data)

        # write data
        outfilename = str(i) + '_converted.tdat'
        outpath = OUT_FOLDER + '/' + outfilename

        with open(outpath, 'w') as out_file:
            out_file.write(out_data_str)







