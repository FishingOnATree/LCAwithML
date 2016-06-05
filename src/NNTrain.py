from __future__ import print_function
import LCUtil
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils



x, y = LCUtil.load_mapped_feature()
x_scale = preprocessing.scale(x)

# use pre-saved random seeds to ensure the same train/cv/test set
random_seeds = LCUtil.load_random_seeds()

# need to normalize mean and standardization
x_train, x_cv, x_test, y_train, y_cv, y_test = \
    LCUtil.separate_training_data(x_scale, y, 0.6, 0.2, random_seeds)
print("Data size: Training, CV, Test = %d, %d, %d" % (x_train.shape[0], x_cv.shape[0], x_test.shape[0]))


model = Sequential()

model.add(Dense(input_dim=39, output_dim=300))
model.add(Activation('tanh'))
model.add(Dropout(0.25))
model.add(Dense(input_dim=300, output_dim=300))
model.add(Activation('tanh'))
model.add(Dropout(0.25))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

batch_size = 32
nb_classes = 10
nb_epoch = 30

model.fit(x_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(x_test, y_test),
          shuffle=True)