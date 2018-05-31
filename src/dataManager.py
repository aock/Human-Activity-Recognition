import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import glob
import traceback
import logging
import sys
from tqdm import tqdm

DATADIR = 'WISDM_RAW'

FILENAME = 'WISDM_RAW/WISDM_raw.txt'

class DataManager():

    def __init__(self, step=20,
                random_seed=42,
                n_features = 3,
                test_size = 0.2,
                n_time_steps = 200,
                datafolder='data'):

        self.step = step
        self.random_seed = random_seed
        self.n_features = n_features
        self.test_size = test_size
        self.n_time_steps = n_time_steps
        self.datafolder = datafolder
        self.columns = ['activity', 'x-axis', 'y-axis', 'z-axis']

        self.LABELS = {
                'NoStairs' : 0,
                'Stairs' : 1
            }

        self.LABEL_COUNTER = {
            0 : 0.0,
            1 : 0.0
        }

    def one_hot(self, num):
        return [int(i == num) for i in range(len(self.LABELS))]

    def load_all(self):

        X_train = None
        X_test = None
        y_train = None
        y_test = None

        self.LABEL_COUNTER = {
                        0 : 0.0,
                        1 : 0.0
                    }

        files = [filename for filename in glob.iglob(self.datafolder + '/**/*.txt', recursive=True)]

        for filename in tqdm(files):
            # print("read data from %s ..." % filename)
            df = pd.read_csv(filename, header = None, names = self.columns)

            segments = []
            labels = []

            # print("number of samples found: %d" % len(df))
            # print("reorganize data...")


            for i in range(0, len(df) - self.n_time_steps, self.step):
                xs = df['x-axis'].values[i: i + self.n_time_steps]
                ys = df['y-axis'].values[i: i + self.n_time_steps]
                zs = df['z-axis'].values[i: i + self.n_time_steps]
                try:
                    label = stats.mode(df['activity'][i: i + self.n_time_steps])[0][0]

                    if label not in self.LABELS:
                        continue
                    else:
                        num = self.LABELS[label]
                        self.LABEL_COUNTER[num] += 1.0
                        label = self.one_hot(num)
                except Exception as e:
                    print(e)
                    print('error in line ' + str(i) + '. skipping...')
                    continue
                segments.append([xs, ys, zs])
                labels.append(label)

            reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, self.n_time_steps, self.n_features)
            labels = np.asarray(labels, dtype = np.float32)
            # print(reshaped_segments.shape)
            # print(labels.shape)

            X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
                reshaped_segments, labels, test_size=self.test_size, random_state=self.random_seed)

            if X_train_tmp.shape[0] > 0:
                if X_train is None:
                    X_train = X_train_tmp
                    X_test = X_test_tmp
                    y_train = y_train_tmp
                    y_test = y_test_tmp
                else:
                    np.concatenate((X_train, X_train_tmp), axis=0)
                    np.concatenate((X_test, X_test_tmp), axis=0)
                    np.concatenate((y_train, y_train_tmp), axis=0)
                    np.concatenate((y_test, y_test_tmp), axis=0)

        return X_train, X_test, y_train, y_test, self.LABEL_COUNTER

if __name__ == "__main__":
    dm = DataManager()
    X_train, X_test, y_train, y_test, label_counter = dm.load_all()

    print(X_train.shape)

    print('data loaded.')
