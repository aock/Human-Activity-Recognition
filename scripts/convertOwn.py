import sys
import glob
import ntpath
import re
import numpy as np
ntpath.basename("a/b/c")

IN_FOLDER = 'import/OWN'
OUT_FOLDER = 'export/OWN'

OUT_TEMPLATE = './' + OUT_FOLDER + '/OWN_%d.tdat'

def gen_filename(i):
    return OUT_TEMPLATE % i

def join(l, sep):
    out_str = ''
    for i, el in enumerate(l):
        out_str += '%s%s' % (el, sep)
    return out_str[:-len(sep)]

def get_2d_list_slice(matrix, start_row, end_row, start_col, end_col):
    return [row[start_col:end_col] for row in matrix[start_row:end_row]]


# def convert(in_data):

#     # define minimum time for a stair walk
#     sec_stair_min = 8
#     timesteps_stair_min = sec_stair_min * 50
#     # define maximum time for a stair walk
#     sec_stair_max = 20
#     timesteps_stair_max = sec_stair_max * 50

#     # duration of unsaftiness
#     dur_unsafe = timesteps_stair_max - timesteps_stair_min


#     out_data = []

#     last_down = 0
#     last_up = 0


#     for i,row in enumerate(in_data):

#         entry = ['NoStairs', row[1], row[2], row[3] ]

#         # downstairs!
#         if (row[-1] != last_down) or (row[-2] != last_up):

#             last_down = row[-1]
#             last_up = row[-2]

#             entry[0] = 'Stairs'

#             for j in range(timesteps_stair_min):
#                 if len(out_data) > j:
#                     out_data[-1 - j][0] = 'Stairs'

#             ids_unsafe = [ ( len(out_data) - timesteps_stair_min - i - 1) for i in range(dur_unsafe)]

#             for id_unsafe in ids_unsafe:
#                 out_data[id_unsafe][0] = 'Unsafe'


#             out_data = fillUnsafe(out_data, ids_unsafe)

#             # change last n out_data labels
#             # what is n? shouldnt be to high or to low

#         out_data.append(entry)

#     return out_data

def dataToString(data):
    filestr = ''
    for i,row in enumerate(data):
        filestr += join(row, ',') + '\n'
    return filestr


if __name__ == "__main__":
    files = [filename for filename in glob.iglob(IN_FOLDER + '/**/*.csv', recursive=True)]

    for i,filepath in enumerate(files):

        label = ""
        if "stair" in filepath:
            print("stair: " + filepath)
            label = "Stairs"
        else:
            print("nostair: " + filepath)
            label = "NoStairs"

        # read and convert data
        in_data = []
        with open(filepath, 'r') as in_file:
            for line in in_file:
                ll = line.strip().split(',')
                entry = [label, float(ll[1]), float(ll[2]), float(ll[3])]
                print(entry)
                in_data.append(entry)

        out_data_str = dataToString(in_data)

        outfilename = gen_filename(i)

        with open(outfilename, 'w') as out_file:
            out_file.write(out_data_str)

