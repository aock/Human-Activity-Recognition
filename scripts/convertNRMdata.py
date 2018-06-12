from src.saveFunctions import *
import sys
import glob
import ntpath
import re
import numpy as np
ntpath.basename("a/b/c")


IN_FOLDER = 'import'
OUT_FOLDER = 'export'

OUT_TEMPLATE = './' + OUT_FOLDER + '/NRM_%d.tdat'

# model for prediction
modelname = 'model2000'
model = readArch(modelname)
readWeights(model, modelname)


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

def predict(data):

    t_data = np.asarray([data], dtype=np.float32)

    res = model.predict(t_data)
    return np.argmax(res)

def get_2d_list_slice(matrix, start_row, end_row, start_col, end_col):
    return [row[start_col:end_col] for row in matrix[start_row:end_row]]

def fillUnsafe(in_data, ids):
    # pseudo:
    # take last unsafe entry -> current entry
    # if next entry is NoStairs:
    # No Stairs -> current entry
    # else
    # predict 200 last data with rnn
    # if Precition is Stairs
    # Stairs -> current entry
    # else
    # NoStairs -> current entry

    ids.sort(reverse=True)

    for id in ids:
        if len(in_data) <= id + 1:
            print("error id out of range")
        else:
            next_entry = in_data[id + 1]
            curr_entry = in_data[id]

            if next_entry[0] == 'Unsafe':
                in_data[id][0] = 'Unsafe'
            elif next_entry[0] == 'NoStairs':
                in_data[id][0] = 'NoStairs'
            else:
                # predict
                t_data = get_2d_list_slice(in_data, id - 200 + 1 , id + 1, 1, 4)
                p_res = predict(t_data)
                if p_res == 1:
                    in_data[id][0] = 'Stairs'
                    print('model predicted additional possible stair sequence')
                else:
                    in_data[id][0] = 'Unsafe'

    return in_data

def convert(in_data):

    # define minimum time for a stair walk
    sec_stair_min = 8
    timesteps_stair_min = sec_stair_min * 50
    # define maximum time for a stair walk
    sec_stair_max = 20
    timesteps_stair_max = sec_stair_max * 50

    # duration of unsaftiness
    dur_unsafe = timesteps_stair_max - timesteps_stair_min


    out_data = []

    last_down = 0
    last_up = 0


    for i,row in enumerate(in_data):

        entry = ['NoStairs', row[1], row[2], row[3] ]

        # downstairs!
        if (row[-1] != last_down) or (row[-2] != last_up):

            last_down = row[-1]
            last_up = row[-2]

            entry[0] = 'Stairs'

            for j in range(timesteps_stair_min):
                if len(out_data) > j:
                    out_data[-1 - j][0] = 'Stairs'

            ids_unsafe = [ ( len(out_data) - timesteps_stair_min - i - 1) for i in range(dur_unsafe)]

            for id_unsafe in ids_unsafe:
                out_data[id_unsafe][0] = 'Unsafe'


            out_data = fillUnsafe(out_data, ids_unsafe)

            # change last n out_data labels
            # what is n? shouldnt be to high or to low

        out_data.append(entry)

    return out_data

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
                ll = line.split(',')
                entry = [float(ll[1]), float(ll[2]), float(ll[3]), float(ll[4]), int(ll[5]), int(ll[6]) ]
                in_data.append(entry)

        # convert data
        out_data = convert(in_data)
        out_data_str = dataToString(out_data)

        # write data
        filename = path_leaf(filepath)
        m = prog.match(filename)
        first = m.group(1)
        second = m.group(2)
        outfilename = filename.replace(first, 'NRM').replace(second, 'tdat')
        print(outfilename)
        outpath = OUT_FOLDER + '/' + outfilename

        with open(outpath, 'w') as out_file:
            out_file.write(out_data_str)







