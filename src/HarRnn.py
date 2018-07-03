import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, ELU, Activation
from keras.layers import Conv1D, LocallyConnected1D, MaxPooling1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization
from keras.layers import add, Input, Flatten
from keras.layers import concatenate
from keras.layers.core import Dense, Dropout
from keras import optimizers, callbacks, regularizers
import keras
import json
import sys
from pprint import pprint

# fix for error in build in keras Elu class
class Elu(ELU):
    def __init__(self, **kwargs):
        self.__name__ = "ELU"
        super(Elu, self).__init__(**kwargs)

"""
Class for Human-activity-Recognition Recurrent neural network.
Use in code:
...
hn = HarRnn()
model = hn.gen()
...
"""
class HarRnn():

    model = None
    optimizer = None
    config = None

    def __init__(self,
                 config={'batch_size': 1024,
                        'seq_size': 200,
                        'network': {
                            'type': 'seq',
                            'reg': False,
                            'rnn':[64,32],
                            'dnn':[8,2]
                        },
                        'optimizer': {
                            'name': 'adam',
                            'lr': 0.0025
                        }
                    },
                 random_seed = 42):

        self.config = config

    def gen(self):
        """
        Generates the complete model, optimizer. Compiles the model
        """

        self.updateModel()
        self.updateOptimizer()
        self.compileModel()

    def updateModel(self):
        self.seqCRNN()

    def seqCRNN(self):
        """
        Generates sequential crnn model from config specifications.
        """


        timesteps = self.config['timesteps']
        input_dim = self.config['input_dim']

        network = self.config['network']

        reg = True
        if 'reg' in network and network['reg'] == False:
            reg = False

        input = Input(shape=(timesteps, input_dim))

        # generate CNN from config
        cnn_out = None

        if 'cnn' in network:
            cnn_filters = self.config['network']['cnn']
            cnn_out = self.cnn(input, cnn_filters, reg)
        else:
            cnn_out = input

        # generate RNN from config
        rnn_out = None
        if 'rnn' in network:
            rnn_size = self.config['network']['rnn']
            rnn_out = self.rnn(cnn_out, rnn_size, reg)
        else:
            rnn_out = cnn_out

        # generate DNN from config
        dnn_out = None
        if 'dnn' in network:
            dnn_size = self.config['network']['dnn']
            dnn_out = self.dnn(rnn_out, dnn_size, reg)
        else:
            dnn_out = rnn_out

        out = Activation('softmax')(dnn_out)

        self.model = keras.models.Model(inputs=[input], outputs=out)

    def rnn(self, input, rnn_size=[64,32], reg=True):
        """generating rnn from list

        Arguments:
            input {keras.layer} -- output from last layer as input of rnn

        Keyword Arguments:
            rnn_size {list} -- layer sizes (default: {[64,32]})
            reg {bool} -- regularize layers (default: {True})

        Returns:
            keras.layer -- output layer of rnn
        """


        lstm_l = [input]
        for i in range(len(rnn_size)-1):
            lstm_l.append(LSTM(rnn_size[i],
                            return_sequences=True)
                            (lstm_l[-1]))

        if reg:
            lstm_l.append(LSTM(rnn_size[-1], recurrent_regularizer=regularizers.l1(0.01))(lstm_l[-1]) )
            lstm_l.append(Dropout(0.5)(lstm_l[-1]) )
        else:
            lstm_l.append(LSTM(rnn_size[-1])(lstm_l[-1]) )

        return lstm_l[-1]

    def dnn(self, input, n_hidden=[32,2], reg=True):
        """ generate deep neural network from layer sizes

        Arguments:
            input {keras.layer} -- last output layer -> input layer of dnn

        Keyword Arguments:
            n_hidden {list} -- [description] (default: {[32,2]})
            reg {bool} -- [description] (default: {True})

        Returns:
            keras.layer -- output layer of dnn
        """

        dnn_l = [input]

        for i in range(len(n_hidden)-1):
            dnn_l.append( Dense(n_hidden[i])(dnn_l[-1]) )
            dnn_l.append( ELU(alpha=0.1)(dnn_l[-1]))
            if reg:
                dnn_l.append( Dropout(0.5)(dnn_l[-1]))

        dnn_l.append(Dense(n_hidden[-1])(dnn_l[-1]) )

        return dnn_l[-1]

    def cnn(self, input, filters=[(32,3),(32,3),(64,3),(64,3)], reg=True):
        """[summary]

        Arguments:
            input {keras.layer} -- last output layer of nn -> input layer of cnn

        Keyword Arguments:
            filters {list} -- list of number of filters and strides. if number of filters eq 0 -> max pooling (default: {[(32,3),(32,3),(64,3),(64,3)]})
            reg {bool} -- regularize convolutional layers (default: {True})

        Returns:
            keras.layer -- output layer of cnn
        """

        cnn_l = [input]

        for i in range(len(filters)):
            if filters[i][0] > 0:
                cnn_l.append(Conv1D(filters[i][0], filters[i][1])(cnn_l[-1]))
                cnn_l.append(Activation('relu')(cnn_l[-1]))
            else:
                cnn_l.append(MaxPooling1D(filters[i][1])(cnn_l[-1]))

        # build model
        return cnn_l[-1]

    def updateOptimizer(self):
        """
        generating and updating optimizer
        """

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
            self.optimizer = optimizers.SGD(lr=opt_cfg['lr'],momentum=opt_cfg['momentum'],decay=opt_cfg['decay'],nesterov=opt_cfg['nesterov'], clipnorm=1.0)

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
        """
        compile model for training
        """

        self.model.compile(loss='categorical_crossentropy',
                  optimizer=self.optimizer,
                  metrics=['accuracy'])

    def getModel(self):
        """get compiled model

        Returns:
            keras.model -- the compiled model
        """

        return self.model
