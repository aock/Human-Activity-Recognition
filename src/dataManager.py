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

    def load_prepared_data_yield(self):

        files = [filename for filename in glob.iglob(self.datafolder + '/*.npz', recursive=True)]

        print("found %d files" % len(files))
        for file in tqdm(files):

            npzfile = np.load(file)

            print(npzfile.files)

            reshaped_segments = npzfile['x']
            labels = npzfile['y']
            print(reshaped_segments.shape)
            print(labels.shape)

            yield reshaped_segments,labels

        return

    def load_prepared_data(self):

        print("load prepared data...")

        it_prep = self.load_prepared_data_yield()
        X,y = next(it_prep)

        for X_tmp, y_tmp in it_prep:
            print("read shape")
            print(X_tmp.shape)
            print(y_tmp.shape)
            X = np.concatenate( (X,X_tmp) )
            y = np.concatenate( (y,y_tmp) )

        return X,y

    def train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed)
        return X_train, X_test, y_train, y_test

    def load_file(self, filename):

        X_train = None
        X_test = None
        y_train = None
        y_test = None

        self.LABEL_COUNTER = {
                        0 : 0.0,
                        1 : 0.0
                    }

        df = pd.read_csv(filename, header = None, names = self.columns)

        segments = []
        labels = []

        for i in range(0, len(df) - self.n_time_steps, self.step):
            xs = df['x-axis'].values[i: i + self.n_time_steps]
            ys = df['y-axis'].values[i: i + self.n_time_steps]
            zs = df['z-axis'].values[i: i + self.n_time_steps]
            xs,ys,zs = self.butter_filter(xs,ys,zs)

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
            labels.append(label)
            if addMulti:
                for j in range(self.additionalAddSmall):
                    segments.append([xs, ys, zs])
                    labels.append(label)


        reshaped_segments = np.asarray(segments, dtype= np.float32)
        reshaped_segments = reshaped_segments.transpose((0,2,1))
        labels = np.asarray(labels, dtype = np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            reshaped_segments, labels, test_size=self.test_size, random_state=self.random_seed)

        return X_train, X_test, y_train, y_test, self.LABEL_COUNTER

    def load_test_file(self, filename):
        X_train = None
        X_test = None
        y_train = None
        y_test = None

        self.LABEL_COUNTER = {
                        0 : 0.0,
                        1 : 0.0
                    }

        df = pd.read_csv(filename, header = None, names = self.columns)

        segments = []
        labels = []

        for i in range(0, len(df) - self.n_time_steps, self.step):
            xs = df['x-axis'].values[i: i + self.n_time_steps]
            ys = df['y-axis'].values[i: i + self.n_time_steps]
            zs = df['z-axis'].values[i: i + self.n_time_steps]
            xs,ys,zs = self.butter_filter(xs,ys,zs)

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
            labels.append(label)
            if addMulti:
                for j in range(self.additionalAddSmall):
                    segments.append([xs, ys, zs])
                    labels.append(label)


        X = np.asarray(segments, dtype= np.float32).transpose((0,2,1))
        y = np.asarray(labels, dtype = np.float32)

        return X,y,len(df)

    def one_hot(self, num):
        return [int(i == num) for i in range(len(self.LABELS))]

    def butter_filter(self,xs,ys,zs):
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
        return xs_l,ys_l,zs_l

    def load_all(self):

        X_train = None
        X_test = None
        y_train = None
        y_test = None

        self.LABEL_COUNTER = {
                        0 : 0.0,
                        1 : 0.0
                    }

        # raw files

        files = [filename for filename in glob.iglob(self.datafolder + '/**/*.tdat', recursive=True)]

        segments = []
        labels = []

        for filename in tqdm(files):
            # print("read data from %s ..." % filename)
            df = pd.read_csv(filename, header = None, names = self.columns)



            # print("number of samples found: %d" % len(df))
            # print("reorganize data...")


            for i in range(0, len(df) - self.n_time_steps, self.step):
                xs = df['x-axis'].values[i: i + self.n_time_steps]
                ys = df['y-axis'].values[i: i + self.n_time_steps]
                zs = df['z-axis'].values[i: i + self.n_time_steps]
                xs,ys,zs = self.butter_filter(xs,ys,zs)

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
                labels.append(label)
                if addMulti:
                    for j in range(self.additionalAddSmall):
                        segments.append([xs, ys, zs])
                        labels.append(label)


        reshaped_segments = np.asarray(segments, dtype= np.float32)
        reshaped_segments = reshaped_segments.transpose((0,2,1))
        labels = np.asarray(labels, dtype = np.float32)
                # print(reshaped_segments.shape)
                # print(labels.shape)

        X_train, X_test, y_train, y_test = train_test_split(
                    reshaped_segments, labels, test_size=self.test_size, random_state=self.random_seed)

        return X_train, X_test, y_train, y_test, self.LABEL_COUNTER


    def prepare_data(self, batch_size=100000):

        self.LABEL_COUNTER = {
                        0 : 0.0,
                        1 : 0.0
                    }

        files = [filename for filename in glob.iglob(self.datafolder + '/**/*.tdat', recursive=True)]
        shuffle(files)
        old_num_data_collected = 0
        num_data_collected = 0

        segments = []
        labels = []

        for filename in files:
            df = pd.read_csv(filename, header = None, names = self.columns)

            # shifting sequence
            for i in range(0, len(df) - self.n_time_steps, self.step):
                old_num_data_collected = num_data_collected
                # data in sequence
                xs = df['x-axis'].values[i: i + self.n_time_steps]
                ys = df['y-axis'].values[i: i + self.n_time_steps]
                zs = df['z-axis'].values[i: i + self.n_time_steps]
                # filter

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

                label_num = 0
                # label stuff (one hot)
                try:
                    label = stats.mode(df['activity'][i: i + self.n_time_steps])[0][0]

                    if label not in self.LABELS:
                        continue
                    else:
                        label_num = self.LABELS[label]
                        label = self.one_hot(label_num)
                except Exception as e:
                    print(e)
                    print('error in line ' + str(i) + '. skipping...')
                    continue

                # adding to buffer
                segments.append([xs_l,ys_l,zs_l])
                labels.append(label)
                self.LABEL_COUNTER[label_num] += 1.0

                if len(labels) == batch_size:
                    yield self.convert_lists(segments, labels)
                    segments = []
                    labels = []

                if label_num == 1:
                    for j in range(self.additionalAddSmall):
                        segments.append([xs_l,ys_l,zs_l])
                        labels.append(label)
                        self.LABEL_COUNTER[label_num] += 1.0

                        if len(labels) == batch_size:
                            yield self.convert_lists(segments, labels)
                            segments = []
                            labels = []

        if len(segments) > 0:
            yield self.convert_lists(segments, labels)
            segments = []
            labels = []

        return

    def get_next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size


        while(True):
            self.LABEL_COUNTER = {
                            0 : 0.0,
                            1 : 0.0
                        }

            files = [filename for filename in glob.iglob(self.datafolder + '/**/*.tdat', recursive=True)]
            shuffle(files)
            old_num_data_collected = 0
            num_data_collected = 0

            segments = []
            labels = []

            for filename in files:
                df = pd.read_csv(filename, header = None, names = self.columns)

                # shifting sequence
                for i in range(0, len(df) - self.n_time_steps, self.step):
                    old_num_data_collected = num_data_collected
                    # data in sequence
                    xs = df['x-axis'].values[i: i + self.n_time_steps]
                    ys = df['y-axis'].values[i: i + self.n_time_steps]
                    zs = df['z-axis'].values[i: i + self.n_time_steps]
                    # filter

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

                    label_num = 0
                    # label stuff (one hot)
                    try:
                        label = stats.mode(df['activity'][i: i + self.n_time_steps])[0][0]

                        if label not in self.LABELS:
                            continue
                        else:
                            label_num = self.LABELS[label]
                            label = self.one_hot(label_num)
                    except Exception as e:
                        print(e)
                        print('error in line ' + str(i) + '. skipping...')
                        continue

                    # adding to buffer
                    segments.append([xs_l,ys_l,zs_l])
                    labels.append(label)
                    self.LABEL_COUNTER[label_num] += 1.0

                    if len(labels) == batch_size:
                        yield self.convert_lists(segments, labels)
                        segments = []
                        labels = []

                    if label_num == 1:
                        for j in range(self.additionalAddSmall):
                            segments.append([xs_l,ys_l,zs_l])
                            labels.append(label)
                            self.LABEL_COUNTER[label_num] += 1.0

                            if len(labels) == batch_size:
                                yield self.convert_lists(segments, labels)
                                segments = []
                                labels = []

    def load_random_gen(self, num_data=100000):
        batch_size = num_data

        self.LABEL_COUNTER = {
                        0 : 0.0,
                        1 : 0.0
                    }

        files = [filename for filename in glob.iglob(self.datafolder + '/**/*.tdat', recursive=True)]
        shuffle(files)
        num_data_collected = 0

        segments = []
        labels = []

        with tqdm(total=num_data) as pbar:

            for filename in files:
                df = pd.read_csv(filename, header = None, names = self.columns)

                # shifting sequence
                for i in range(0, len(df) - self.n_time_steps, self.step):
                    num_data_collected = len(lables)
                    # data in sequence
                    xs = df['x-axis'].values[i: i + self.n_time_steps]
                    ys = df['y-axis'].values[i: i + self.n_time_steps]
                    zs = df['z-axis'].values[i: i + self.n_time_steps]
                    # filter

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

                    label_num = 0
                    # label stuff (one hot)
                    try:
                        label = stats.mode(df['activity'][i: i + self.n_time_steps])[0][0]

                        if label not in self.LABELS:
                            continue
                        else:
                            label_num = self.LABELS[label]
                            label = self.one_hot(label_num)
                    except Exception as e:
                        print(e)
                        print('error in line ' + str(i) + '. skipping...')
                        continue

                    # adding to buffer
                    segments.append([xs_l,ys_l,zs_l])
                    labels.append(label)
                    self.LABEL_COUNTER[label_num] += 1.0
                    pbar.update(1)

                    if len(labels) == batch_size:
                        X,y = self.convert_lists(segments, labels)
                        X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=self.test_size )
                        yield X_train, X_test, y_train, y_test
                        segments = []
                        labels = []

                    if label_num == 1:
                        for j in range(self.additionalAddSmall):
                            segments.append([xs_l,ys_l,zs_l])
                            labels.append(label)
                            self.LABEL_COUNTER[label_num] += 1.0

                            if len(labels) == batch_size:
                                X,y = self.convert_lists(segments, labels)
                                X_train, X_test, y_train, y_test = train_test_split(
                                         X, y, test_size=self.test_size )
                                yield X_train, X_test, y_train, y_test
                                segments = []
                                labels = []

                            pbar.update(1)

    def convert_lists(self, segments, labels):
        reshaped_segments = np.asarray(segments, dtype= np.float32)
        reshaped_segments = reshaped_segments.transpose((0,2,1))
        labels = np.asarray(labels, dtype = np.float32)
        return reshaped_segments, labels


class DataGenerator():

    def __init__(self, step=20,
                test_size = 0.2,
                n_time_steps = 200,
                datafolder='data',
                filter = {
                    'butter_low':{
                        'order' : 6,
                        'fs' : 30.0,
                        'cutoff': 3.667
                    }
                },
                additionalAddSmall=0,
                batch_size=1024):

        self.filter = filter
        self.step = step
        self.test_size = test_size
        self.n_time_steps = n_time_steps
        self.datafolder = datafolder
        self.additionalAddSmall = additionalAddSmall
        self.batch_size = batch_size
        self.columns = ['activity', 'x-axis', 'y-axis', 'z-axis']

        self.LABELS = {
                'NoStairs' : 0,
                'Stairs' : 1
            }

        self.LABEL_COUNTER = {
            0 : 0.0,
            1 : 0.0
        }

        self.num_d = -1

    def get_next_batch(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        while(True):
            self.LABEL_COUNTER = {
                            0 : 0.0,
                            1 : 0.0
                        }

            files = [filename for filename in glob.iglob(self.datafolder + '/**/*.tdat', recursive=True)]
            shuffle(files)
            old_num_data_collected = 0
            num_data_collected = 0

            segments = []
            labels = []

            for filename in files:
                df = pd.read_csv(filename, header = None, names = self.columns)

                # shifting sequence
                for i in range(0, len(df) - self.n_time_steps, self.step):
                    old_num_data_collected = num_data_collected
                    # data in sequence
                    xs = df['x-axis'].values[i: i + self.n_time_steps]
                    ys = df['y-axis'].values[i: i + self.n_time_steps]
                    zs = df['z-axis'].values[i: i + self.n_time_steps]
                    # filter
                    xs,ys,zs = self.filter_data(xs,ys,zs)

                    label_num = 0
                    # label stuff (one hot)
                    try:
                        label = stats.mode(df['activity'][i: i + self.n_time_steps])[0][0]

                        if label not in self.LABELS:
                            continue
                        else:
                            label_num = self.LABELS[label]
                            label = self.one_hot(label_num)
                    except Exception as e:
                        print(e)
                        print('error in line ' + str(i) + '. skipping...')
                        continue

                    # adding to buffer
                    segments.append([xs,ys,zs])
                    labels.append(label)
                    self.LABEL_COUNTER[label_num] += 1.0

                    if len(labels) == batch_size:
                        yield self.convert_lists(segments, labels)
                        segments = []
                        labels = []

                    if label_num == 1:
                        for j in range(self.additionalAddSmall):
                            segments.append([xs,ys,zs])
                            labels.append(label)
                            self.LABEL_COUNTER[label_num] += 1.0

                            if len(labels) == batch_size:
                                yield self.convert_lists(segments, labels)
                                segments = []
                                labels = []

    def convert_lists(self, segments, labels):
        reshaped_segments = np.asarray(segments, dtype= np.float32)
        reshaped_segments = reshaped_segments.transpose((0,2,1))
        labels = np.asarray(labels, dtype = np.float32)
        return reshaped_segments, labels

    def filter_data(self,xs,ys,zs):
        xs_ret = xs
        ys_ret = ys
        zs_ret = zs
        if 'butter_low' in self.filter:
            xs_ret, ys_ret, zs_ret = self.butter_filter(xs_ret,ys_ret,zs_ret)

        return xs_ret,ys_ret,zs_ret

    def butter_filter(self,xs,ys,zs):
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
        return xs_l,ys_l,zs_l

    def one_hot(self, num):
        return [int(i == num) for i in range(len(self.LABELS))]

    def count_data(self):

        if self.num_d == -1:
            # takes a long time
            self.num_d = 0

            files = [filename for filename in glob.iglob(self.datafolder + '/**/*.tdat', recursive=True)]

            for filename in files:
                df = pd.read_csv(filename, header = None, names = self.columns)

                # shifting sequence
                for i in range(0, len(df) - self.n_time_steps, self.step):
                    label_num = 0
                    # label stuff (one hot)
                    try:
                        label = stats.mode(df['activity'][i: i + self.n_time_steps])[0][0]

                        if label not in self.LABELS:
                            continue
                        else:
                            label_num = self.LABELS[label]
                            label = self.one_hot(label_num)
                    except Exception as e:
                        print(e)
                        print('error in line ' + str(i) + '. skipping...')
                        continue

                    # counting
                    self.num_d += 1
                    if label_num == 1:
                        for j in range(self.additionalAddSmall):
                            self.num_d += 1
        else:
            # takes short time
            return self.num_d


if __name__ == "__main__":
    # dm = DataManager()
    # X_train, X_test, y_train, y_test, label_counter = dm.load_all()

    dg = DataGenerator()

    count = dg.count_data()
    print(count)

    # train_data_gen = dg.get_next_batch(batch_size=1024)
    # val_data_gen = dg.get_next_batch(batch_size=256)

    # for X,y in train_data_gen:
    #     print(X.shape)

    print('data loaded.')
