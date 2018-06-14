import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process Options.')
    parser.add_argument('-f','--file', type=str, help='path to analysis file', required=True)
    args = parser.parse_args()

    filename = args.file
    data = []

    with open(filename,'r') as data_file:
        for line in data_file:
            line = line.strip().split(',')
            entry = [float(line[0]), float(line[1])]
            data.append( np.asarray(entry) )

    data = np.asarray(data)
    num_data = data.shape[0]
    x = np.arange(0,num_data)
    loss = data[:,0]
    acc = data[:,1]

    plt.plot(x,loss,'r', label='loss')
    plt.plot(x,acc,'b', label='acc')
    plt.xlabel('training days')
    plt.ylabel('performance')
    plt.legend()
    plt.show()