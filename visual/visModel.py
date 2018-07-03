from keras import backend as K
from keras.utils import plot_model
from src.saveFunctions import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Options.')
    parser.add_argument('-m','--model', type=str, help='model name', required=True)
    args = parser.parse_args()

    model = readArch(args.model)
    readWeights(model, args.model)
    plot_model(model, to_file='%s.png' % args.model, show_layer_names=False, show_shapes=True)
    print(model.summary())