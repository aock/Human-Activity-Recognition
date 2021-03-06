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
                        help='export directory name',
                        required=False)
    parser.add_argument('-s','--size',
                        default=100000,
                        type=int,
                        help='num samples per file',
                        required=False)
    parser.add_argument('--step',
                        default=5,
                        type=int,
                        help='step of interval',
                        required=False)
    args = parser.parse_args()

    dm = DataManager(datafolder=args.data, test_size=0.0, step=args.step)

    if not os.path.exists(args.export):
        os.makedirs(args.export)

    counter = 0


    for counter,(X,y) in enumerate(dm.prepare_data(batch_size=args.size)):
        print(counter)
        np.savez(args.export + '/train_' + str(counter), x=X,y=y)

    print("exported %d files" % counter)

