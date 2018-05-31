import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, ELU, Activation
from keras.layers.core import Dense, Dropout
from keras import optimizers, callbacks, regularizers
import json
import sys
from pprint import pprint

label_weights = np.array([1/100, 30])
w = K.variable(value=label_weights, dtype='float32', name='loss_weights')

def meanSquaredWeightedError(target, output):
    return K.mean( K.square(w * (target - output) ) )

class Elu(ELU):
    def __init__(self, **kwargs):
        self.__name__ = "ELU"
        super(Elu, self).__init__(**kwargs)

class HarRnn():

    model = None
    optimizer = None
    config = None

    def __init__(self,
                 config={'batch_size': 1024,
                        'seq_size': 200,
                        'n_hidden': 64,
                        'dataset': 'WISDM',
                        'optimizer': {
                            'name': 'rmsprop'
                        }
                    },
                 debug = True,
                 random_seed = 42):

        self.config = config

        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

        if debug:
            # for reproducibility
            # https://github.com/fchollet/keras/issues/2280



            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1
            )


            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)

    def gen(self):
        self.updateModel()
        self.updateOptimizer()
        self.compileModel()

    def updateModel(self):
        # collect information

        n_hidden = self.config['n_hidden']
        timesteps = self.config['timesteps']
        input_dim = self.config['input_dim']
        n_classes = self.config['n_classes']

        # build model
        elu = Elu(alpha=1.0)

        self.model = Sequential()
        self.model.add(LSTM(n_hidden, return_sequences=True, input_shape=(timesteps, input_dim)))
        self.model.add(LSTM(n_hidden, recurrent_regularizer=regularizers.l1(0.01)))

        self.model.add(Dropout(0.5))

        # self.model.add(Dense(n_hidden,
        #                 kernel_regularizer=regularizers.l2(0.01),
        #                 activity_regularizer=regularizers.l1(0.01)))
        self.model.add(Dense(n_hidden)) # war vorher
        self.model.add(ELU(alpha=1.0))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(n_classes, activation='softmax' ))

        # Old model
        # self.model.add(Dense(n_hidden)) # linear activation
        # self.model.add(Dense(n_classes, activation='softmax' )) # softmax acitivation

    def updateOptimizer(self):
        opt_opt = self.config['optimizer']

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
            self.optimizer = optimizers.RMSprop(lr=opt_cfg['lr'],rho=opt_cfg['rho'],epsilon=opt_cfg['epsilon'],decay=opt_cfg['decay'])

        if opt_opt['name'] == 'sgd':
            opt_cfg = {
                'lr':0.01,
                'decay':0.0,
                'momentum':0.0,
                'nesterov':False
            }
            opt_cfg.update(opt_opt)
            pprint(opt_cfg)
            self.optimizer = optimizers.SGD(lr=opt_cfg['lr'],momentum=opt_cfg['momentum'],decay=opt_cfg['decay'],nesterov=opt_cfg['nesterov'])

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
            self.optimizer = optimizers.Adam(
                                    lr=opt_cfg['lr'],
                                    beta_1=opt_cfg['beta_1'],
                                    beta_2=opt_cfg['beta_2'],
                                    epsilon=opt_cfg['epsilon'],
                                    decay=opt_cfg['decay'],
                                    amsgrad=opt_cfg['amsgrad']
                                )

    def updateLossFunction(self):
        pass

    def compileModel(self):

        self.model.compile(loss='categorical_crossentropy',
                  optimizer=self.optimizer,
                  metrics=['accuracy'])
        # self.model.compile(loss=meanSquaredWeightedError,
        #           optimizer=self.optimizer,
        #           metrics=['accuracy', meanSquaredWeightedError])

    def getModel(self):
        return self.model
