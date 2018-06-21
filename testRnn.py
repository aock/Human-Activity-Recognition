import argparse
from src.saveFunctions import *
import json
import sys
from pprint import pprint
from dataWISDM import load_data
from src.dataManager import DataManager
import numpy as np
import tensorflowjs as tfjs

from utils import confusion_matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process Options.')
    parser.add_argument('-m','--model', type=str, help='model name', required=True)
    parser.add_argument('-d','--sampleDir', type=str, help='path to dir with samples (.tdat)', required=False)
    parser.add_argument('-f','--sampleFile', type=str, help='path to file with samples (.tdat)', required=False)
    parser.add_argument('-l','--labelFile', type=str, help='path to json label file.', required=True )
    parser.add_argument('-e','--export', type=str, help='model name to export model as json', required=False)

    args = parser.parse_args()

    model = readArch(args.model)
    readWeights(model, args.model)
    sampleDir = args.sampleDir
    labelFile = args.labelFile

    labels = []

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
    elif args.sampleFile:
        X, _, Y, _, class_counter = dm.load_file(args.sampleFile)
    else:
        print("ERROR: file or direction required for test (-f or -d)")


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

    if args.export:
        print("exporting tfjs model...")
        tfjs.converters.save_keras_model(model, args.export)

