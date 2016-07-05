__author__ = 'Rays'
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD
from sklearn import svm


class ModelBase:

    def __init__(self):
        pass

    def train(self, x_train, y_train, x_cv, y_cv):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)

    def predict(self, x):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)

    @staticmethod
    def get_param_headers():
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)


# class NeuralNetworkModel(ModelBase):
#
#     @property
#     def __init__(self, settings):
#         self.model = Sequential()
#         # model variables
#         self.batch_size = settings["batch_size"]
#         self.nb_epoch = settings["nb_epoch"]
#         self.hidden_unit_width = settings["hidden_unit_width"]
#         self.drop_out_rate = settings["drop_out_rate"]
#
#     @property
#     def train(self, x_train, y_train, x_cv, y_cv):
#         self.model.add(Dense(input_dim=x_train.shape[1], output_dim=self.hidden_unit_width))
#         self.model.add(Activation('relu'))
#         self.model.add(Dropout(self.drop_out_rate))
#         self.model.add(Dense(input_dim=self.hidden_unit_width, output_dim=self.hidden_unit_width))
#         self.model.add(Activation('relu'))
#         self.model.add(Dropout(self.drop_out_rate))
#         self.model.add(Dense(input_dim=self.hidden_unit_width, output_dim=self.hidden_unit_width))
#         self.model.add(Activation('relu'))
#         self.model.add(Dropout(self.drop_out_rate))
#         self.model.add(Dense(output_dim=2))
#         self.model.add(Activation('softmax'))
#         # let's train the model using SGD + momentum (how original).
#         sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
#         self.model.compile(loss='sparse_categorical_crossentropy',
#                            optimizer=sgd,
#                            metrics=['accuracy'])
#         self.model.fit(x_train, y_train,
#                        batch_size=self.batch_size,
#                        nb_epoch=self.nb_epoch,
#                        validation_data=(x_cv, y_cv),
#                        shuffle=True)
#
#     @property
#     def predict(self, x):
#         result = []
#         h = self.model.predict(x)
#         for r in h:
#             if r[0] > 0.5:
#                 result.append(0)
#             else:
#                 result.append(1)
#         print(result)
#         return result
#
#     @staticmethod
#     def get_param_headers():
#         return ["nb_epoch", "batch_size", "hidden_unit_width", "drop_out_rate"]


class SVMModel(ModelBase):

    def __init__(self, settings):
        # C: small --> soft-margin and allows misclassification (regularization)
        # gamma: inverse to influential area of SVs, small gamma --> each SV only affects a small area and cannot capture the curve.
        # self.model = svm.SVC(C=settings["C"],
        #                      max_iter=settings["max_iter"],
        #                      cache_size=settings["cache_size"],
        #                      gamma=settings["gamma"],
        #                      class_weight=settings["class_weight"])
        # [C=1, class_weights={0: 0.88, 1: 0.12}] is by far the best indicator. RS 16/07/05
        self.model = svm.SVC(kernel="rbf",  C=settings["C"],
                             class_weight=settings["class_weight"], gamma=settings["gamma"])
        pass

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def get_param_headers(self):
        return ["C", "max_iter", "class_weight", "gamma"]


MODEL_DICT = { #"NN": NeuralNetworkModel,
              "SVM": SVMModel}


def get_instance(model_name, settings):
    if not MODEL_DICT.has_key(model_name):
        raise ValueError("%s is not a valid options in %s" % (model_name, str(MODEL_DICT.keys())))
    else:
        model_class = MODEL_DICT[model_name]
        return model_class(settings)