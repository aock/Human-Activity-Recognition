from src.dataManager import DataManager
import numpy as np
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process Options.')
    parser.add_argument('-d','--data',
                        default='data',
                        type=str,
                        help='path to dir with samples (.tdat)',
                        required=False)
    parser.add_argument('-e','--export',
                        default='preparedData',
                        type=str,
                        help='model name to export model as json',
                        required=False)
    args = parser.parse_args()

    dm = DataManager(datafolder=args.data, test_size=0.0, step=1)

    X, _, y, _, class_counter = dm.load_all()

    if not os.path.exists(args.export):
        os.makedirs(args.export)

    np.save(args.export + '/train', X)
    np.save(args.export + '/labels',y)
