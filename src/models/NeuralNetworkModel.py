from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

from models import ModelBase


class NeuralNetworkModel(ModelBase.ModelBase):

    def __init__(self, settings):
        self.model = Sequential()
        # model variables
        self.batch_size = settings["batch_size"]
        self.nb_epoch = settings["nb_epoch"]
        self.hidden_unit_width = settings["hidden_unit_width"]
        self.drop_out_rate = settings["drop_out_rate"]

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.add(Dense(input_dim=x_train.shape[1], output_dim=self.hidden_unit_width))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.drop_out_rate))
        self.model.add(Dense(input_dim=self.hidden_unit_width, output_dim=self.hidden_unit_width))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.drop_out_rate))
        self.model.add(Dense(input_dim=self.hidden_unit_width, output_dim=self.hidden_unit_width))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.drop_out_rate))
        self.model.add(Dense(output_dim=2))
        self.model.add(Activation('softmax'))
    #    model.add(Dense(output_dim=10))
    #    model.add(Activation('softmax'))
        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       nb_epoch=self.nb_epoch,
                       validation_data=(x_cv, y_cv),
                       shuffle=True)

    def predict(self, x):
        #TODO: Fix the predict code to return a mx1 prediction result
        return self.model.predict(x)