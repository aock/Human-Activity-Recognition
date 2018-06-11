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
                        'n_hidden': [64,64,64,64],
                        'dataset': 'WISDM',
                        'optimizer': {
                            'name': 'rmsprop'
                        }
                    },
                 debug = True,
                 random_seed = 42):

        self.config = config

        # np.random.seed(random_seed)
        # tf.set_random_seed(random_seed)

        if debug:
            # for reproducibility
            # https://github.com/fchollet/keras/issues/2280



            # session_conf = tf.ConfigProto(
            #     intra_op_parallelism_threads=1,
            #     inter_op_parallelism_threads=1
            # )


            sess = tf.Session(graph=tf.get_default_graph())
            K.set_session(sess)

    def gen(self):
        self.updateModel()
        self.updateOptimizer()
        self.compileModel()

    def updateModel(self):
        #self.concatCRNN()
        self.seqCRNN()
        
    def seqCRNN(self):
        
        timesteps = self.config['timesteps']
        input_dim = self.config['input_dim']

        network = self.config["network"]

        reg = True
        if "reg" in network and network["reg"] == False:
            reg = False

        input = Input(shape=(timesteps, input_dim))
 
        # CNN
        cnn_out = None

        if "cnn" in network:
            cnn_filters = self.config["network"]["cnn"]
            cnn_out = self.cnn(input, cnn_filters, reg)
        else:
            cnn_out = input
        
        # RNN after CNN 
        rnn_out = None
        if "rnn" in network:
            rnn_size = self.config["network"]["rnn"]
            rnn_out = self.rnn(cnn_out, rnn_size, reg)
        else:
            rnn_out = cnn_out
        
        # DNN after RNN
        dnn_out = None
        if "dnn" in network:
            dnn_size = self.config["network"]["dnn"]
            dnn_out = self.dnn(rnn_out, dnn_size, reg)
        else:
            dnn_out = rnn_out

        
        out = Activation("softmax")(dnn_out)

        self.model = keras.models.Model(inputs=[input], outputs=out)


    def concatCRNN(self):
        n_hidden = self.config['n_hidden']
        timesteps = self.config['timesteps']
        input_dim = self.config['input_dim']
        n_classes = self.config['n_classes']
         
        input = Input(shape=(timesteps, input_dim))

        # RNN
        rnn_size = [64,64]
        rnn_out = self.rnn(input, rnn_size)
        
        # DNN after RNN
        rdnn_size = [32,8]
        rdnn_out = self.dnn(rnn_out, rdnn_size)

        # CNN
        cnn_filters = [(64,3),(32,3),(0,3)]
        cnn_out = self.cnn(input, cnn_filters)
        
        # DNN after RNN
        cdnn_size = [32,8]
        cdnn_out = Flatten()(self.dnn(cnn_out, cdnn_size))

        # merge
        concat = concatenate([rdnn_out,cdnn_out])

        # dnn after concat
        dnn_size = [16,16,2]
        dnn_out = self.dnn(concat, dnn_size)


        self.model = keras.models.Model(inputs=[input], outputs=dnn_out)


    def rnn(self, input, rnn_size=[64,32], reg=True):
        # build model

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
        dnn_l = [input]

        for i in range(len(n_hidden)-1):
            dnn_l.append( Dense(n_hidden[i])(dnn_l[-1]) )
            dnn_l.append( ELU(alpha=0.1)(dnn_l[-1]))
            if reg:
                dnn_l.append( Dropout(0.5)(dnn_l[-1]))

        dnn_l.append(Dense(n_hidden[-1])(dnn_l[-1]) )

        return dnn_l[-1]

    def cnn(self, input, filters=[(32,3),(32,3),(64,3),(64,3)], reg=True):
        # collect information

        cnn_l = [input]

        for i in range(len(filters)):
            if filters[i][0] > 0:
                cnn_l.append(Conv1D(filters[i][0], filters[i][1])(cnn_l[-1]))
                cnn_l.append(Activation('relu')(cnn_l[-1]))
            else:
                cnn_l.append(MaxPooling1D(filters[i][1])(cnn_l[-1]))

        # build model
        return cnn_l[-1]

    def rcnn(self):
        # collect information

        n_hidden = self.config['n_hidden']
        timesteps = self.config['timesteps']
        input_dim = self.config['input_dim']
        n_classes = self.config['n_classes']


        # build model
        elu = Elu(alpha=1.0)

        self.model = Sequential()
        self.model.add(Conv1D(32, 5, input_shape=(timesteps, input_dim) ))
        #self.model.add(BatchNormalization())
        #self.model.add(Activation('relu'))
        #self.model.add(Conv1D(32, 3, activation='relu'))
        #self.model.add(MaxPooling1D(3))
        self.model.add(LSTM(64, return_sequences=True)) 
        self.model.add(LSTM(32, recurrent_regularizer=regularizers.l1(0.01)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(8) ) # war vorher
        self.model.add(ELU(alpha=1.0))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='softmax' ))

    def rcnn2(self):
        # collect information

        n_hidden = self.config['n_hidden']
        timesteps = self.config['timesteps']
        input_dim = self.config['input_dim']
        n_classes = self.config['n_classes']


        # build model
        elu = Elu(alpha=1.0)

        input = keras.layers.Input(shape=(timesteps, input_dim))
        # splitted network 1: input -> cnn -> cnn_out_3
        cnn_out_1 = Conv1D(32, 3, input_shape=(timesteps, input_dim ), activation='relu')(input)
        cnn_out_2 = Conv1D(32, 3, activation='relu')(cnn_out_1)
        cnn_out_3 = MaxPooling1D(3)(cnn_out_2)
        cnn_out_4 = Conv1D(64, 3, activation='relu')(cnn_out_3)
        cnn_out_5 = Conv1D(64, 3, activation='relu')(cnn_out_4)
        cnn_out_6 = keras.layers.Flatten()(cnn_out_5)

        # splitted network 2: input -> rnn -> rnn_out
        lstm_out_1 = LSTM(64, return_sequences=True)(input)
        lstm_out_2 = LSTM(64, recurrent_regularizer=regularizers.l1(0.01))(lstm_out_1)
        lstm_out_3 = Dropout(0.5)(lstm_out_2)

        # merge rnn and cnn
        added = keras.layers.concatenate([cnn_out_6, lstm_out_3])

        # two fully connected layers with regularization
        out_1 = Dense(32)(added)
        out_2 = ELU(alpha=1.0)(out_1)
        out_3 = Dropout(0.1)(out_2)
        out_4 = Dense(n_classes, activation='softmax' )(out_3)
        self.model = keras.models.Model(inputs=[input], outputs=out_4)

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

        self.model.compile(loss='categorical_crossentropy',
                  optimizer=self.optimizer,
                  metrics=['accuracy'])
        # self.model.compile(loss=meanSquaredWeightedError,
        #           optimizer=self.optimizer,
        #           metrics=['accuracy', meanSquaredWeightedError])

    def getModel(self):
        return self.model
