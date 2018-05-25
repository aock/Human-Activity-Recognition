import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)

import tensorflowjs as tfjs
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras import optimizers
from keras import callbacks
import json
import sys
from pprint import pprint

def create_rnn_model( options={} ):
   # TODO: implement
   pass


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
            

if __name__ == "__main__":
    # parse options
    
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
    
    if len(sys.argv) > 1:
        config_file = str(sys.argv[1])
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


    use_wisdm_dataset = True

    if use_wisdm_dataset:
        from dataWISDM import load_data
    else:
        from data import load_data

    from utils import confusion_matrix

    epochs = config['epochs']
    batch_size = config['batch_size']
    n_hidden = config['n_hidden']

    def _count_classes(y):
        return len(set([tuple(category) for category in y]))

    X_train, X_test, Y_train, Y_test = load_data()

    if use_wisdm_dataset:
        # WISDM dataset
        timesteps = X_train.shape[1]
        input_dim = X_train.shape[2]
        n_classes = Y_train.shape[1]
    else:
        # HAR Dataset
        timesteps = len(X_train[0])
        input_dim = len(X_train[0][0])
        n_classes = _count_classes(Y_train)
    
    # create model

    model = Sequential()
    model.add(LSTM(n_hidden, return_sequences=True, input_shape=(timesteps, input_dim)))
    model.add(LSTM(n_hidden))
    model.add(Dropout(0.5))
    model.add(Dense(n_hidden))
    model.add(Dense(n_classes, activation='softmax'))

    optimizer = None
    opt_opt = config['optimizer']

    print('Using Optimizer: ' + opt_opt['name'])
    # rmsprop
    if opt_opt['name'] == 'rmsprop':
        # keras defaults
        opt_cfg = {
            'lr':0.001,
            'rho':0.9,
            'epsilon':None,
            'decay':0.0
        }
        opt_cfg.update(opt_opt)
        pprint(opt_cfg)
        optimizer = optimizers.RMSprop(lr=opt_cfg['lr'],rho=opt_cfg['rho'],epsilon=opt_cfg['epsilon'],decay=opt_cfg['decay'])

    if opt_opt['name'] == 'sgd':
        opt_cfg = {
            'lr':0.01,
            'decay':0.0,
            'momentum':0.0,
            'nesterov':False
        }
        opt_cfg.update(opt_opt)
        pprint(opt_cfg)
        optimizer = optimizers.SGD(lr=opt_cfg['lr'],momentum=opt_cfg['momentum'],decay=opt_cfg['decay'],nesterov=opt_cfg['nesterov'])
    
    if opt_opt['name'] == 'adam':
        opt_cfg = {
            'lr' : 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': None,
            'decay': 0.0,
            'amsgrad': False
        }
        opt_cfg.update(opt_opt)
        pprint(opt_cfg)
        optimizer = optimizers.Adam(
                                 lr=opt_cfg['lr'],
                                 beta_1=opt_cfg['beta_1'],
                                 beta_2=opt_cfg['beta_2'],
                                 epsilon=opt_cfg['epsilon'],
                                 decay=opt_cfg['decay'],
                                 amsgrad=opt_cfg['amsgrad']
                               )

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    early_stopping = EarlyStoppingValAcc(value=0.95, verbose=1)

    model.fit(X_train,
              Y_train,
              batch_size=batch_size,
              validation_data=(X_test, Y_test),
              epochs=epochs,
              callbacks=[early_stopping])

    # Evaluate
    tfjs.converters.save_keras_model(model, ".")
    print(confusion_matrix(Y_test, model.predict(X_test)))
