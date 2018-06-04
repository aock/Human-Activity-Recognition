
IN_FILE = './pretraining_data/data3.txt'
OUT_TEMPLATE = './export/WISDM_ar/WISDM_at_%d.tdat'

def gen_filename(i):
    return OUT_TEMPLATE % i

def join(l, sep):
    out_str = ''
    for i, el in enumerate(l):
        out_str += '%s%s' % (el, sep)
    return out_str[:-len(sep)]

if __name__ == "__main__":

    with open(IN_FILE, 'r') as in_file:
        fc = 0
        curr_id = 0
        file_str = ''
        for line in in_file:

            ll = line.split(',')

            id = int(ll[0])

            if id != curr_id:

                out_filename = gen_filename(fc)
                with open(out_filename, 'w') as out_file:
                    out_file.write(file_str)

                file_str = ''
                fc += 1
                curr_id = id

            ll = [ll[1],ll[3],ll[4],ll[5]]
            conv_str = join(ll,',')
            file_str += conv_str

        # remaining lines
        if file_str != '':
            out_filename = gen_filename(fc)
            with open(out_filename, 'w') as out_file:
                out_file.write(file_str)
