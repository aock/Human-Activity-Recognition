import argparse
from src.saveFunctions import *
import json
import sys
from pprint import pprint
from dataWISDM import load_data
from src.dataManager import DataManager
import numpy as np
import tensorflowjs as tfjs
import matplotlib.pyplot as plt

from utils import confusion_matrix

def most_common(lst):
    return max(set(lst), key=lst.count)

def percentage(lst):

    res_lst = []

    for i in range(len(lst) ):
        percent = 0.0
        for elem in lst[i]:
            percent += float(elem)

        percent /= float( len(lst[i]) )
        res_lst.append(percent)

    return res_lst


def count_ascend(lst):
    ascend_cnt = 0
    old = 0
    for v in lst:
        if v == 1 and v != old:
            ascend_cnt += 1
            old = 1
        elif v == 0 and v != old:
            old = 0
    return ascend_cnt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process Options.')
    parser.add_argument('-m','--model', type=str, help='model name', required=True)
    parser.add_argument('-d','--sampleDir', type=str, help='path to dir with samples (.tdat)', required=False)
    parser.add_argument('-f','--sampleFile', type=str, help='path to file with samples (.tdat)', required=False)
    parser.add_argument('-l','--labelFile', type=str, help='path to json label file.', required=False )
    parser.add_argument('-e','--export', type=str, help='model name to export model as json', required=False)

    args = parser.parse_args()

    model = readArch(args.model)
    readWeights(model, args.model)
    sampleDir = args.sampleDir


    labels = ["NoStairs","Stairs"]


    if args.labelFile:
        labelFile = args.labelFile
        print('loading available labels: ' + labelFile)
        with open(labelFile) as f:
            data = json.load(f)
            labels = data['labels']

    print("labels found:")
    pprint(labels)

    dm = DataManager(datafolder=sampleDir, test_size=0.0)

    X = None
    Y = None
    class_counter = None
    if args.sampleDir:
        X, _, Y, _, class_counter = dm.load_all()
        print('class_counter: ')
        pprint(class_counter)

        pred = 0
        try:
            pred = model.predict(X)
        except:
            a = 0

        pred_am = np.argmax(pred, axis=1)
        Y_am = np.argmax(Y, axis=1)
        # for i in range(pred_am.shape[0]):
        #     print(str(Y_am[i]) + " == " + str(pred_am[i]) )

        # total success ratio
        equal = np.sum(pred_am == Y_am)
        ratio_total = equal / pred.shape[0]
        print(ratio_total)

        print(confusion_matrix(Y, pred))
    elif args.sampleFile:
        X, y, num_acc = dm.load_test_file(args.sampleFile)


        print(num_acc)
        pred = 0
        try:
            pred = model.predict(X)
        except:
            a = 0

        num_seq = X.shape[0]
        seq_len = X.shape[1]

        label_mat = []

        print(X.shape)
        print(pred.shape)

        # prediction human readable
        pred_hr = np.argmax(pred, axis=1)

        pred_stairs = pred[:,1]

        print(pred_hr.shape)

        timesteps = 20

        c = 0

        interval = [0,0]

        num_acc_predicted = seq_len + timesteps * (num_seq - 1)

        for i in range( num_acc_predicted ):
            label_mat.append([])

        for i_local,i_global in enumerate(range(seq_len,num_acc,timesteps)):
            # update interval
            interval[0] = i_global-seq_len
            interval[1] = i_global

            for i in range(interval[0],interval[1]):
                label_mat[i].append(pred_hr[i_local])

        percent_labels = percentage(label_mat)
        percent_labels = np.asarray(percent_labels)

        real_labels = []
        for l in label_mat:
            mc = most_common(l)
            real_labels.append( mc )

        stairs = count_ascend(real_labels)

        t = np.arange(0, num_acc_predicted )
        real_labels = np.asarray(real_labels)

        plt.figure(1)
        plt.subplot(211)
        axes = plt.gca()
        axes.set_ylim([-0.1,1.1])
        plt.plot(t, real_labels, '.', label='%d stairs' % stairs)
        plt.plot(t, percent_labels, '-', label='percentage' )
        plt.legend()

        plt.subplot(212)
        axes = plt.gca()
        axes.set_ylim([-0.1,1.1])
        plt.plot(np.arange(0,pred_stairs.shape[0]), pred_stairs, '-', label='prediction stairs' )

        plt.legend()
        plt.show()







    else:
        print("ERROR: file or direction required for test (-f or -d)")




    if args.export:
        print("exporting tfjs model...")
        tfjs.converters.save_keras_model(model, args.export)

