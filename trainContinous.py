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
from src.dataManager import DataManager
import argparse
import os
from src.saveFunctions import *
import math


def create_class_weight(labels_dict, mu=0.15):
    total = sum(labels_dict.values())
    class_weight = dict()

    for key, value in labels_dict.items():
        v = float(value)
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

def join(l, sep):
    out_str = ''
    for i, el in enumerate(l):
        out_str += '%s%s' % (el, sep)
    return out_str[:-len(sep)]

if __name__ == "__main__":
    # parse options

    parser = argparse.ArgumentParser(description='Process Options.')
    parser.add_argument('-c','--config', type=str, help='config json file path')
    parser.add_argument('-m','--model', type=str, help='update model: specify folfer containing weights.h5 and arch.json', required=True)
    parser.add_argument('-d','--dataDir', type=str, help='path to training data (.tdat)', required=False)
    parser.add_argument('--analysisFile', type=str, help='path to analysis file', required=False)

    args = parser.parse_args()

    config = {
                'epochs': 60,
                'batch_size': 1024,
                'seq_size': 200,
                'n_hidden': [64,64,64,64],
                'dataset': 'WISDM',
                'optimizer': {
                    'name': 'rmsprop'
                },
                'additionalAddSmall': 0
             }

    if args.config:
        config_file = args.config
        print('loading config: ' + config_file)
        with open(config_file) as f:
            data = json.load(f)
            config.update(data)


    sess = tf.Session(graph=tf.get_default_graph())
    K.set_session(sess)

    ###################################
    ####### data handling start
    ##################################

    datafolder = 'data'

    if args.dataDir:
        datafolder = args.dataDir

    dm = DataManager(datafolder=datafolder, additionalAddSmall=config['additionalAddSmall'])

    model = None

    training_counter = 1

    data_per_day = 100000

    if "data_per_day" in config:
        data_per_day = config["data_per_day"]


    try:
        hr = HarRnn(config=config, debug=True)
        hr.updateOptimizer()

        model = readArch(args.model)
        readWeights(model, args.model)

        model.compile(loss='categorical_crossentropy',
                optimizer=hr.optimizer,
                metrics=['accuracy'])
        print("Existing model loaded")
    except:
        hr = HarRnn(config=config, debug=True)
        hr.gen()
        model = hr.getModel()
        print("Train not existing model")
        print(model.summary())



    analysisFile = ""
    if args.analysisFile:
        analysisFile = args.model + '/' + args.analysisFile
        try:
            os.remove(args.analysisFile)
        except OSError:
            pass

    while(True):

        print("starting training day: " + str(training_counter))
        print("data samples to learn: " + str(data_per_day))

        X_train, X_test, Y_train, Y_test, class_counter = dm.load_random(num_data=data_per_day)
        #TODO: predict Y_train save loss and acc in file


        # load new when stair label is not existing
        while(class_counter[1] < 5):
            X_train, X_test, Y_train, Y_test, class_counter = dm.load_random(num_data=data_per_day)


        # WISDM dataset
        print("INPUT SHAPE")
        print(X_train.shape)
        config['timesteps'] = X_train.shape[1]
        config['input_dim'] = X_train.shape[2]
        config['n_classes'] = Y_train.shape[1]
        print('number of different labels found: %d' % config['n_classes'])

        print('class_counter: ')
        pprint(class_counter)
        ####################################
        ###### model creation #############
        ####################################

        cw = None
        if 'class_weight' in config:
            if config['class_weight'] != 'auto':
                print("using predefined class weights")
                cw = config['class_weight']
            else:

                cw = create_class_weight(class_counter, mu=1.0)
                print("using automatically determined class weights:")
                pprint(cw)

        if args.analysisFile:

            scores = model.evaluate(X_test, Y_test)
            score_str = join(scores, ',')
            with open(analysisFile, "a") as myfile:
                myfile.write(score_str + '\n')

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


        model.fit(X_train,
                Y_train,
                batch_size=batch_size,
                validation_data=(X_test, Y_test),
                epochs=epochs,
                callbacks=cbacks,
                class_weight=cw)


        ##################################
        ###### updates and saving ########
        ##################################

        print("saving model...")
        saveModel(model, args.model)
        saveArch(model, args.model)
        saveWeights(model, args.model)

        # save tensorflowjs model
        print("exporting tfjs model...")
        tfjs.converters.save_keras_model(model, args.model)

        print(confusion_matrix(Y_test, model.predict(X_test)))
        training_counter += 1
