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
    """
    DataManager for:
    - reading and preparing 'tdat' files in specified directory.
    - reading already prepared 'tdat' files from specified directory.
    Output can be directly used for HarRnn.py generated neural network
    """

    def __init__(self, step=20,
                random_seed=42,
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
        """ Constructor of DataManager

        Keyword Arguments:
            step {int} -- step of shifting interval for sequence generation (default: {20})
            random_seed {int} -- random seed (default: {42})
            test_size {float} -- percentage of test data (default: {0.2})
            n_time_steps {int} -- sequence length in number of timesteps (default: {200})
            datafolder {str} -- data folder for searching 'tdat' files or prepared files (default: {'data'})
            filter {dict} -- filter specification (default: {{'butter_low':{'order' : 6,'fs' : 30.0,'cutoff': 3.667}}})
            additionalAddSmall {int} -- duplicate stair class data (default: {0})
        """

        self.filter = filter

        self.step = step
        self.random_seed = random_seed
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
        """ load prepared data. one yield per file

        Yields:
            np.array -- acceleration sequences
            np.array -- one hot labels
        """

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
        """ load all prepared data

        Returns:
            np.array -- acceleration sequences
            np.array -- one hot labels
        """

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
        """ split data in train and test(validation) data

        Arguments:
            X {np.array} -- sequences of acceleration data
            y {np.array} -- one hot labels

        Returns:
            np.array -- sequences of acceleration data for training
            np.array -- sequences of acceleration data for testing
            np.array -- one hot labels for training
            np.array -- one hot labels for testing
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed)
        return X_train, X_test, y_train, y_test

    def load_file(self, filename):
        """ Loading one specific file

        Arguments:
            filename {string} -- filename to load

        Returns:
            np.array -- sequences of acceleration data for training
            np.array -- sequences of acceleration data for testing
            np.array -- one hot labels for training
            np.array -- one hot labels for testing
            dict -- Label Counter dictionary
        """


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
        """ loading specific test file

        Arguments:
            filename {string} -- testfile name

        Returns:
            np.array -- X: acceleration sequences
            np.array -- y: one hot labels
            int      -- len(df): number of acceleration data in total
        """

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
        """ Generate one hot design from number

        Arguments:
            num {int} -- number of label (0 -> 'NoStairs', 1 -> 'Stairs')

        Returns:
            np.array -- resulting one hot design
        """

        return [int(i == num) for i in range(len(self.LABELS))]

    def butter_filter(self,xs,ys,zs):
        """ butter lowpass filter multidimensional

        Arguments:
            xs {np.array} -- acceleration sequence x
            ys {np.array} -- acceleration sequence y
            zs {np.array} -- acceleration sequence z

        Returns:
            np.array -- xs_l: filtered acceleration x
            np.array -- ys_l: filtered acceleration y
            np.array -- zs_l: filtered acceleration z
        """

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
        """ loading all 'tdat' files of data_directory

        Returns:
            np.array -- sequences of acceleration data for training
            np.array -- sequences of acceleration data for testing
            np.array -- one hot labels for training
            np.array -- one hot labels for testing
            dict -- Label Counter dictionary
        """


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
        """ prepare data into chunks of size batch_size. yields the chunks

        Keyword Arguments:
            batch_size {int} -- chunk size for splitting data (default: {100000})

        Yields:
            np.array -- acceleration sequences of size batch_size (except of last)
            np.array -- one hot labels of size batch_size (except of last)
        """

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
        """ Random data loading with given chunk size. Yield version

        Keyword Arguments:
            batch_size {[type]} -- [description] (default: {None})

        Yields:
            np.array -- acceleration sequences of size batch_size (except of last)
            np.array -- one hot labels of size batch_size (except of last)
        """

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
        """ Random training and test data with given chunk size. Yield version

        Keyword Arguments:
            batch_size {[type]} -- [description] (default: {None})

        Yields:
            np.array -- training acceleration sequences of size batch_size (except of last)
            np.array -- test acceleration sequences of size batch_size (except of last)
            np.array -- training one hot labels of size batch_size (except of last)
            np.array -- test one hot labels of size batch_size (except of last)
        """
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
        """convert acceleration sequences (segments) and labels to np.arrays

        Arguments:
            segments {list} -- list of acceleration sequences
            labels {list} -- list of one hot labels

        Returns:
            np.array -- reshaped_segments: numpy array of sequences
            np.array -- labels: numpy array of one hot labels
        """

        reshaped_segments = np.asarray(segments, dtype= np.float32)
        reshaped_segments = reshaped_segments.transpose((0,2,1))
        labels = np.asarray(labels, dtype = np.float32)
        return reshaped_segments, labels


class DataGenerator():
    """
    DataGenerator for loading data without using too much memory.
    Currently not used
    """

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

    """
    Test of Datamanger. Example usage.
    """

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
