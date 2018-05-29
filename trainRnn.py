import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)

import tensorflowjs as tfjs
from keras import backend as K
from keras import callbacks
import json
import sys
from pprint import pprint
from utils import confusion_matrix
from src.HarRnn import HarRnn
from keras.models import model_from_json, load_model
import argparse
import os

class EarlyStoppingValAcc(callbacks.Callback):
    def __init__(self, value=0.95, verbose=0):
        super(callbacks.Callback, self).__init__()
        self.monitor = 'val_acc'
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
        elif self.verbose > 0:
            print("val_acc: %f <= %f" % (current,self.value))


def saveModel(model, modelname):
    if not os.path.exists(modelname):
        os.makedirs(modelname)
    model.save(modelname + '/entireModel.h5')

def readModel(modelname):
    return load_model(modelname + '/entireModel.h5')

def saveWeights(model, modelname):
    if not os.path.exists(modelname):
        os.makedirs(modelname)
    model.save_weights(modelname + '/weights.h5')

def readWeights(model, modelname):
    model.load_weights(modelname + '/weights.h5')

def saveArch(model, modelname):
    if not os.path.exists(modelname):
        os.makedirs(modelname)
    with open(modelname+'/arch.json', 'w') as f:
        f.write(model.to_json())

def readArch(modelname):
    with open(modelname + '/arch.json', 'r') as f:
        model = model_from_json(f.read())
    return model

if __name__ == "__main__":
    # parse options

    parser = argparse.ArgumentParser(description='Process Options.')
    parser.add_argument('--config', type=str, help='config json file path')
    parser.add_argument('--update', type=str, help='update model: specify folfer containing weights.h5 and arch.json')
    parser.add_argument('--save', type=str, help='model name. Generates folder containing arch.json and weights.h5')
    parser.add_argument('--export', type=str, help='model name. Generates tfjs export json file in dir named by model name')
    args = parser.parse_args()

    config = {
                'epochs': 60,
                'batch_size': 1024,
                'seq_size': 200,
                'n_hidden': 64,
                'dataset': 'WISDM',
                'optimizer': {
                    'name': 'rmsprop'
                }
             }

    if args.config:
        config_file = args.config
        print('loading config: ' + config_file)
        with open(config_file) as f:
            data = json.load(f)
            config.update(data)


    # for reproducibility
    # https://github.com/fchollet/keras/issues/2280
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )


    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    ###################################
    ####### data handling start
    ##################################
    use_wisdm_dataset = True

    if use_wisdm_dataset:
        from dataWISDM import load_data
    else:
        from data import load_data

    def _count_classes(y):
        return len(set([tuple(category) for category in y]))

    X_train, X_test, Y_train, Y_test = load_data()

    if use_wisdm_dataset:
        # WISDM dataset
        config['timesteps'] = X_train.shape[1]
        config['input_dim'] = X_train.shape[2]
        config['n_classes'] = Y_train.shape[1]
    else:
        # HAR Dataset
        config['timesteps'] = len(X_train[0])
        config['input_dim'] = len(X_train[0][0])
        config['n_classes'] = _count_classes(Y_train)

    ####################################
    ###### model creation #############
    ####################################

    model = None

    if args.update:
        model = readModel(args.update)
    else:
        hr = HarRnn(config=config, debug=True, random_seed=42)
        model = hr.getModel()

    ##################################
    ###### training ##################
    ##################################

    cbacks = []

    epochs = config['epochs']
    batch_size = config['batch_size']

    if 'early_val_acc' in config:
        early_val_acc = config['early_val_acc']
        early_stopping = EarlyStoppingValAcc(value=early_val_acc, verbose=1)
        cbacks = [early_stopping]
        print("Early stopping with val acc < %f " % early_val_acc)


    model.fit(X_train,
            Y_train,
            batch_size=batch_size,
            validation_data=(X_test, Y_test),
            epochs=epochs,
            callbacks=cbacks)



    ##################################
    ###### updates and saving ########
    ##################################

    if args.update:
        print("updating model...")
        saveModel(model, args.update)

    if args.save:
        print("saving model...")
        saveModel(model, args.save)

    if args.export:
        # save tensorflowjs model
        print("exporting tfjs model...")
        tfjs.converters.save_keras_model(model, args.export)
        # print(confusion_matrix(Y_test, model.predict(X_test)))
