import numpy as np
# np.random.seed(42)
import tensorflow as tf
# tf.set_random_seed(42)

import tensorflowjs as tfjs
from keras import backend as K
from keras import callbacks
import json
import sys
from pprint import pprint
from utils import confusion_matrix
from src.HarRnn import HarRnn, meanSquaredWeightedError
from src.dataManager import DataManager, DataGenerator
import argparse
import os
from src.saveFunctions import *
import math

# PARAMETERS
USE_DATA_GENERATOR = True


def create_class_weight(labels_dict, mu=0.15):
    total = sum(labels_dict.values())
    class_weight = dict()

    for key, value in labels_dict.items():
        v = float(value)
        if v == 0.0:
            class_weight[key] = 5.0
        else:
            score = math.log(mu*total/v)
            class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

class EarlyStopping(callbacks.Callback):
    def __init__(self, value=0.95, monitor='val_acc',verbose=0, bigger=True):
        super(callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.bigger = bigger

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if self.bigger and current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
        elif not self.bigger and current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
        elif self.verbose > 0:
            if self.bigger:
                print(self.monitor + ": %f <= %f" % (current,self.value))
            else:
                print(self.monitor + ": %f >= %f" % (current,self.value))


if __name__ == "__main__":
    # parse options

    parser = argparse.ArgumentParser(description='Process Options.')
    parser.add_argument('--config', type=str, help='config json file path')
    parser.add_argument('--update', type=str, help='update model: specify folfer containing weights.h5 and arch.json')
    parser.add_argument('--save', type=str, help='model name. Generates folder containing arch.json and weights.h5')
    parser.add_argument('--export', type=str, help='model name. Generates tfjs export json file in dir named by model name')

    parser.add_argument('-d','--dataDir', type=str, help='path to training data (.tdat)', required=False)
    args = parser.parse_args()

    config = {
                'epochs': 60,
                'save_each':10,
                'batch_size': 1024,
                'seq_size': 200,
                'n_hidden': [64,64,64,64],
                'dataset': 'WISDM',
                'optimizer': {
                    'name': 'rmsprop'
                },
                'additionalAddSmall': 0,
                'lowMemory':False
             }

    if args.config:
        config_file = args.config
        print('loading config: ' + config_file)
        with open(config_file) as f:
            data = json.load(f)
            config.update(data)


    if config['lowMemory']:
        USE_DATA_GENERATOR = True
    else:
        USE_DATA_GENERATOR = False

    # # for reproducibility
    # # https://github.com/fchollet/keras/issues/2280
    # session_conf = tf.ConfigProto(
    #     intra_op_parallelism_threads=1,
    #     inter_op_parallelism_threads=1
    # )


    sess = tf.Session(graph=tf.get_default_graph())
    K.set_session(sess)

    ###################################
    ####### data handling start
    ##################################

    datafolder = 'data'

    if args.dataDir:
        datafolder = args.dataDir

    # datamanager for memory high consumption -> fast
    dm = DataManager(datafolder=datafolder, additionalAddSmall=config['additionalAddSmall'])
    # data generator for memory low consumption -> low speed
    dg = DataGenerator(datafolder=datafolder, additionalAddSmall=config['additionalAddSmall'])

    # determine shapes from dataset
    info_gen = dg.get_next_batch(batch_size=1)
    X_train,Y_train = next(info_gen)
    print("INPUT SHAPE")
    print(X_train.shape)
    config['timesteps'] = X_train.shape[1]
    config['input_dim'] = X_train.shape[2]
    config['n_classes'] = Y_train.shape[1]
    print('number of different labels found: %d' % config['n_classes'])

    ####################################
    ###### model creation #############
    ####################################

    model = None
    cw = None
    # if 'class_weight' in config:
    #     if config['class_weight'] != 'auto':
    #         print("using predefined class weights")
    #         cw = config['class_weight']
    #     else:

    #         cw = create_class_weight(class_counter, mu=1.0)
    #         print("using automatically determined class weights:")
    #         pprint(cw)
    cw = {0:1,1:5}

    if args.update:
        hr = HarRnn(config=config, debug=True, random_seed=42)
        hr.updateOptimizer()

        model = readArch(args.update)
        readWeights(model, args.update)

        model.compile(loss='categorical_crossentropy',
                  optimizer=hr.optimizer,
                  metrics=['accuracy'])
    else:
        hr = HarRnn(config=config, debug=True, random_seed=42)
        hr.gen()
        model = hr.getModel()
        print(model.summary())

    ##################################
    ###### training ##################
    ##################################

    cbacks = []

    epochs = config['epochs']
    batch_size = config['batch_size']


    if 'early_stopping' in config:
        early_config = config['early_stopping']

        # check if anything is fine with early config
        interupt = False

        if not 'val' in early_config:
            print('please specify value for early stopping. e.g. 0.1')
            interupt = True

        if not 'monitor' in early_config:
            print('please specify monitor for early stopping. e.g. "val_loss"')
            interupt = True

        if not 'bigger' in early_config:
            print('please specify if early stopping executes for values bigger than specified value. e.g. false for smaller value early stopping')
            interupt = True

        if interupt:
            exit()

        early_val = early_config['val']
        early_monitor = early_config['monitor']
        early_bigger = early_config['bigger']
        early_stopping = EarlyStopping(value=early_val,monitor=early_monitor,bigger=early_bigger, verbose=1)
        cbacks = [early_stopping]
        if early_bigger:
            print("Early stopping with montor %s > %f " % (early_monitor, early_val) )
        else:
            print("Early stopping with montor %s < %f " % (early_monitor, early_val) )

    if USE_DATA_GENERATOR:

        # determine input shape
        train_data_gen = dg.get_next_batch(batch_size=batch_size)
        val_data_gen = dg.get_next_batch(batch_size=batch_size/4)

        model.fit_generator(generator=train_data_gen,
                validation_data=val_data_gen,
                epochs=epochs,
                steps_per_epoch=20,
                validation_steps=20,
                class_weight={0:1,1:5})

    else:
        X_train, X_test, Y_train, Y_test, class_counter = dm.load_all()

        curr_epoch = 0
        save_each = config['save_each']

        for curr_epoch in range(0,epochs,save_each):
            print("current epoch: " + str(curr_epoch))
            model.fit(X_train,
                Y_train,
                batch_size=batch_size,
                validation_data=(X_test, Y_test),
                epochs=save_each,
                callbacks=cbacks,
                class_weight=cw)
            try:
                if args.update:
                    print("updating model...")
                    saveModel(model, args.update)
                    saveArch(model, args.update)
                    saveWeights(model, args.update)

                if args.save:
                    print("saving model...")
                    saveModel(model, args.save)
                    saveArch(model, args.save)
                    saveWeights(model, args.save)

                if args.export:
                    # save tensorflowjs model
                    print("exporting tfjs model...")
                    tfjs.converters.save_keras_model(model, args.export)
            except:
                print("error while saving model")


    ##################################
    ###### updates and saving ########
    ##################################

    if args.update:
        print("updating model...")
        saveModel(model, args.update)
        saveArch(model, args.update)
        saveWeights(model, args.update)

    if args.save:
        print("saving model...")
        saveModel(model, args.save)
        saveArch(model, args.save)
        saveWeights(model, args.save)

    if args.export:
        # save tensorflowjs model
        print("exporting tfjs model...")
        tfjs.converters.save_keras_model(model, args.export)

    print(confusion_matrix(Y_test, model.predict(X_test)))
