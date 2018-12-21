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
                 layer_count=10,
                 layer_type='Covolution1D',
                 filter_size=2,
                 kernel_size=10,
                 input_shape=(48000, 1)
                 ):

        self.model.add(kl.InputLayer(input_shape=input_shape))
        if layer_type == 'Convolution1D':
            for _ in range(layer_count):
                self.model.add(
                    kl.Conv1D(filters=max(filter_size*_, 1), kernel_size=kernel_size, strides=1, activation='relu')
                )
                self.model.add(
                    kl.MaxPool1D(pool_size=2)
                )

            for _ in range(layer_count):
                self.model.add(
                    kl.Conv1D(filters=max(filter_size*_, 1), kernel_size=kernel_size, strides=1, activation='relu')
                )
                self.model.add(
                    kl.UpSampling1D(size=2)
                )

        self.model.optimizer = optmz.SGD()
        self.model.loss = 'mean_squared_error'
        self.model.metrics = 'accuracy'
        self.model.sample_weight_mode = None
        self.model.loss_weights = None

        # Naming the Autoencoder to save and read later
        self.model_name = self.model_name.format('#{}_Layers_{}'.format(layer_count, str(datetime.datetime.now())))

    def train_to_epoch(self, input_data, output_data, epochs=100, batch_size=64):
        self.model.fit(input_data, output_data, epochs=epochs, batch_size=batch_size)

    def train_to_loss(self, input_data, output_data, loss_limit=0.1, batch_size=64):
        t_h = numpy.Inf
        while t_h > loss_limit:
            t_h = self.model.fit(input_data, output_data, epochs=10, batch_size=batch_size).history['loss'][-1]

    def encode_input(self, input_data):
        layer_count = len(self.model.layers)
        encoded = self.model.layers[layer_count//2].output
        functor = ke.backend.function([self.model.layers[0], ke.backend.learning_phase()], [encoded])
        layer_out = functor([input_data, 1.])
        return layer_out

    def save(self):
        self.model.save(model_save_format.format(self.model_name))
