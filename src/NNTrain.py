from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


def train(x_train, y_train, x_cv, y_cv):
    model = Sequential()

    # model variables
    batch_size = 32
    nb_epoch = 100
    hidden_unit_width = 300
    drop_out_rate = 0.25


    model.add(Dense(input_dim=x_train.shape[1], output_dim=hidden_unit_width))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out_rate))
    model.add(Dense(input_dim=hidden_unit_width, output_dim=hidden_unit_width))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out_rate))
    model.add(Dense(input_dim=hidden_unit_width, output_dim=hidden_unit_width))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out_rate))
    model.add(Dense(output_dim=2))
    model.add(Activation('softmax'))
#    model.add(Dense(output_dim=10))
#    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(x_cv, y_cv),
              shuffle=True)
