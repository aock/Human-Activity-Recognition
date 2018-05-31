import argparse
from src.saveFunctions import *
import json
import sys
from pprint import pprint
from dataWISDM import load_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process Options.')
    parser.add_argument('-m','--model', type=str, help='model path', required=True)
    parser.add_argument('-f','--sampleFile', type=str, help='path to file with samples', required=True)
    parser.add_argument('-l','--labelFile', type=str, help='path to json label file.', required=True )

    args = parser.parse_args()

    model = readArch(args.model)
    readWeights(model, args.model)
    sampleFile = args.sampleFile
    labelFile = args.labelFile

    labels = []

    print('loading available labels: ' + labelFile)
    with open(labelFile) as f:
        data = json.load(f)
        labels = data['labels']

    print("labels found:")
    pprint(labels)

    X, _, Y, _, _ = load_data(filename=sampleFile, test_size=0.0)

    pred = model.predict(X)

    print(pred)
    print(Y)
