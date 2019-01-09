from keras import layers as kl
from keras import models as km
from keras import optimizers as optmz

import datetime
import numpy
import keras as ke

from configuration import model_save_format


class Autoencoder:

    model_name = 'AutoEncoder_{}'
    model = km.Sequential()

    def __init__(self,
                 layer_count=10
                 ):
        self.layer_count = layer_count
        self.input_seq = kl.Input(shape=(48000, 1))

        encoder = kl.Conv1D(filters=1, kernel_size=3, padding='same', data_format='channels_last', dilation_rate=1,
                            # encoder is developed here
                            activation='relu', use_bias=False)(self.input_seq)
        encoder = kl.MaxPooling1D(pool_size=(2,))(encoder)
        for _ in range(self.layer_count // 2)[1:]:
            encoder = kl.Conv1D(filters=2 ** _, kernel_size=3, padding='same', data_format='channels_last',
                                dilation_rate=1,
                                activation='relu', use_bias=False)(encoder)
            encoder = kl.MaxPooling1D(pool_size=(2,))(encoder)

        decoder = encoder
        for _ in range(self.layer_count // 2)[::-1]:
            decoder = kl.Conv1D(filters=2 ** _, kernel_size=3, padding='same', data_format='channels_last',
                                dilation_rate=1,
                                activation='relu', use_bias=False)(decoder)
            decoder = kl.UpSampling1D(2)(decoder)

        self.model = ke.Model(self.input_seq, decoder)

        sgd = ke.optimizers.SGD(decay=1e-5, momentum=0.9, nesterov=True)

        self.model.compile(optimizer=sgd, loss='mean_squared_error')

        # Naming the Autoencoder to save and read later
        self.model_name = self.model_name.format('#{}_Layers_{}'.format(layer_count, str(datetime.datetime.now())))

    def train_to_epoch(self, input_data, output_data, epochs=100, batch_size=100):
        return self.model.fit([input_data], [output_data], epochs=epochs, batch_size=batch_size).history['loss'][-1]

    def train_to_loss(self, input_data, output_data, loss_limit=0.1, batch_size=1):
        t_h = numpy.Inf
        while t_h > loss_limit:
            t_h = self.model.fit([input_data], [output_data], epochs=10, batch_size=batch_size).history['loss'][-1]

    def encode_input(self, input_data):
        layer_count = len(self.model.layers)
        encoded = self.model.layers[layer_count//2].output
        functor = ke.backend.function([self.model.layers[0], ke.backend.learning_phase()], [encoded])
        layer_out = functor([input_data, 1.])
        return layer_out

    def load(self, filename):
        self.model.load_weights(filename)

    def save(self):
        self.model.save_weights(model_save_format.format(self.model_name))
