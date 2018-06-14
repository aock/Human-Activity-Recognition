import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import glob
import traceback
import logging
import sys
from tqdm import tqdm
from random import shuffle
from .filters import *

class DataManager():

    def __init__(self, step=20,
                random_seed=42,
                n_features = 3,
                test_size = 0.2,
                n_time_steps = 200,
                datafolder='data',
                filter={
                    'butter_low':{
                        'order' : 6,
                        'fs' : 30.0,
                        'cutoff': 3.667
                    }
                },
                additionalAddSmall=0):

        self.filter = filter

        self.step = step
        self.random_seed = random_seed
        self.n_features = n_features
        self.test_size = test_size
        self.n_time_steps = n_time_steps
        self.datafolder = datafolder
        self.additionalAddSmall = additionalAddSmall
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

        files = [filename for filename in glob.iglob(self.datafolder + '/**/*.tdat', recursive=True)]

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

                addMulti = False

                try:
                    label = stats.mode(df['activity'][i: i + self.n_time_steps])[0][0]

                    # if label == 'Stairs':
                    #     print('Stairs')
                    # else:
                    #     print('-')
                    if label not in self.LABELS:
                        continue
                    else:
                        num = self.LABELS[label]
                        if num == 1:
                            addMulti = True
                            self.LABEL_COUNTER[num] += 1.0 + self.additionalAddSmall
                        else:
                            self.LABEL_COUNTER[num] += 1.0
                        label = self.one_hot(num)
                except Exception as e:
                    print(e)
                    print('error in line ' + str(i) + '. skipping...')
                    continue

                segments.append([xs, ys, zs])

                if 'butter_low' in self.filter:
                    # normalize raw data
                    xs_norm = normalize_mean(xs)
                    ys_norm = normalize_mean(ys)
                    zs_norm = normalize_mean(zs)
                    # butter filter
                    butter_low = self.filter['butter_low']
                    cutoff = butter_low['cutoff']
                    fs = butter_low['fs']
                    order = butter_low['order']
                    xs_l = butter_lowpass_filter(xs_norm, cutoff, fs, order)
                    ys_l = butter_lowpass_filter(ys_norm, cutoff, fs, order)
                    zs_l = butter_lowpass_filter(zs_norm, cutoff, fs, order)
                    # extend features
                    segments[-1].extend( [xs_norm, ys_norm, zs_norm, xs_l, ys_l, zs_l] )

                labels.append(label)
                if addMulti:
                    for j in range(self.additionalAddSmall):
                        segments.append([xs,ys,zs])

                        if 'butter_low' in self.filter:
                            # normalize raw data
                            xs_norm = normalize_mean(xs)
                            ys_norm = normalize_mean(ys)
                            zs_norm = normalize_mean(zs)
                            # butter filter
                            butter_low = self.filter['butter_low']
                            cutoff = butter_low['cutoff']
                            fs = butter_low['fs']
                            order = butter_low['order']
                            xs_l = butter_lowpass_filter(xs_norm, cutoff, fs, order)
                            ys_l = butter_lowpass_filter(ys_norm, cutoff, fs, order)
                            zs_l = butter_lowpass_filter(zs_norm, cutoff, fs, order)
                            # extend features
                            segments[-1].extend( [xs_norm, ys_norm, zs_norm, xs_l, ys_l, zs_l] )

                        labels.append(label)


            reshaped_segments = np.asarray(segments, dtype= np.float32)
            if reshaped_segments.shape[0] > 0:

                reshaped_segments = reshaped_segments.transpose((0,2,1))
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
                        X_train = np.concatenate((X_train, X_train_tmp), axis=0)
                        X_test = np.concatenate((X_test, X_test_tmp), axis=0)
                        y_train = np.concatenate((y_train, y_train_tmp), axis=0)
                        y_test = np.concatenate((y_test, y_test_tmp), axis=0)

        return X_train, X_test, y_train, y_test, self.LABEL_COUNTER

    def load_random(self, num_data=100000):
        X_train = None
        X_test = None
        y_train = None
        y_test = None

        self.LABEL_COUNTER = {
                        0 : 0.0,
                        1 : 0.0
                    }

        files = [filename for filename in glob.iglob(self.datafolder + '/**/*.tdat', recursive=True)]

        shuffle(files)

        old_num_data_collected = 0
        num_data_collected = 0



        with tqdm(total=num_data) as pbar:

            interupt = False

            for filename in files:

                print(filename)

                if interupt:
                    break

                # print("read data from %s ..." % filename)
                df = pd.read_csv(filename, header = None, names = self.columns)

                segments = []
                labels = []

                # print("number of samples found: %d" % len(df))
                # print("reorganize data...")


                for i in range(0, len(df) - self.n_time_steps, self.step):

                    if interupt:
                        break

                    old_num_data_collected = num_data_collected

                    xs = df['x-axis'].values[i: i + self.n_time_steps]
                    ys = df['y-axis'].values[i: i + self.n_time_steps]
                    zs = df['z-axis'].values[i: i + self.n_time_steps]

                    addMulti = False

                    try:
                        label = stats.mode(df['activity'][i: i + self.n_time_steps])[0][0]

                        # if label == 'Stairs':
                        #     print('Stairs')
                        # else:
                        #     print('-')
                        if label not in self.LABELS:
                            continue
                        else:
                            num = self.LABELS[label]
                            if num == 1:
                                addMulti = True
                                self.LABEL_COUNTER[num] += 1.0 + self.additionalAddSmall
                            else:
                                self.LABEL_COUNTER[num] += 1.0
                            label = self.one_hot(num)
                    except Exception as e:
                        print(e)
                        print('error in line ' + str(i) + '. skipping...')
                        continue

                    segments.append([xs,ys,zs])

                    if 'butter_low' in self.filter:
                        # normalize raw data
                        xs_norm = normalize_mean(xs)
                        ys_norm = normalize_mean(ys)
                        zs_norm = normalize_mean(zs)
                        # butter filter
                        butter_low = self.filter['butter_low']
                        cutoff = butter_low['cutoff']
                        fs = butter_low['fs']
                        order = butter_low['order']
                        xs_l = butter_lowpass_filter(xs_norm, cutoff, fs, order)
                        ys_l = butter_lowpass_filter(ys_norm, cutoff, fs, order)
                        zs_l = butter_lowpass_filter(zs_norm, cutoff, fs, order)
                        # extend features
                        segments[-1].extend( [xs_norm, ys_norm, zs_norm, xs_l, ys_l, zs_l] )

                    labels.append(label)
                    if addMulti:
                        for j in range(self.additionalAddSmall):
                            segments.append([xs,ys,zs])

                            if 'butter_low' in self.filter:
                                # normalize raw data
                                xs_norm = normalize_mean(xs)
                                ys_norm = normalize_mean(ys)
                                zs_norm = normalize_mean(zs)
                                # butter filter
                                butter_low = self.filter['butter_low']
                                cutoff = butter_low['cutoff']
                                fs = butter_low['fs']
                                order = butter_low['order']
                                xs_l = butter_lowpass_filter(xs_norm, cutoff, fs, order)
                                ys_l = butter_lowpass_filter(ys_norm, cutoff, fs, order)
                                zs_l = butter_lowpass_filter(zs_norm, cutoff, fs, order)
                                # extend features
                                segments[-1].extend( [xs_norm, ys_norm, zs_norm, xs_l, ys_l, zs_l] )

                            labels.append(label)

                    # correct
                    #print(segments[-1])

                    num_data_collected = self.LABEL_COUNTER[0] + self.LABEL_COUNTER[1]

                    if num_data_collected > num_data:
                        interupt = True
                        break
                    else:
                        pbar.update(num_data_collected - old_num_data_collected)


                reshaped_segments = np.asarray(segments, dtype= np.float32)
                if reshaped_segments.shape[0] > 0:

                    reshaped_segments = reshaped_segments.transpose((0,2,1))
                    labels = np.asarray(labels, dtype = np.float32)

                    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
                        reshaped_segments, labels, test_size=self.test_size, random_state=self.random_seed)

                    if X_train_tmp.shape[0] > 0:
                        if X_train is None:
                            X_train = X_train_tmp
                            X_test = X_test_tmp
                            y_train = y_train_tmp
                            y_test = y_test_tmp
                        else:
                            X_train = np.concatenate((X_train, X_train_tmp), axis=0)
                            X_test = np.concatenate((X_test, X_test_tmp), axis=0)
                            y_train = np.concatenate((y_train, y_train_tmp), axis=0)
                            y_test = np.concatenate((y_test, y_test_tmp), axis=0)

        return X_train, X_test, y_train, y_test, self.LABEL_COUNTER


if __name__ == "__main__":
    dm = DataManager()
    X_train, X_test, y_train, y_test, label_counter = dm.load_all()

    print(X_train.shape)

    print('data loaded.')
