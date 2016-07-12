__author__ = 'Rays'
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD
import numpy as np
from sklearn import svm, datasets, feature_selection, cross_validation
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


class ModelBase:

    def __init__(self, settings):
        self.model = self.get_model(settings)
        self.settings = settings

    def get_model(self, settings):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)

    def train(self, x_train, y_train, x_cv, y_cv):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)

    def predict(self, x):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)

    def plot_feature_selection_diagram(self, x, y):
        clf = Pipeline([('selector', self.get_feature_selection()), ('learning', self.model)])
        score_means = list()
        score_stds = list()
        percentiles = (10, 20, 30, 40, 50, 60, 70)
        for percentile in percentiles:
            clf.set_params(selector__percentile=percentile)
            # Compute cross-validation score using all CPUs
            this_scores = cross_validation.cross_val_score(clf, x, y, n_jobs=1)
            score_means.append(this_scores.mean())
            score_stds.append(this_scores.std())
            print("Finished percentile %d" % percentile)
        plt.errorbar(percentiles, score_means, np.array(score_stds))

        plt.title('Performance of the SVM-Anova varying the percentile of features selected')
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')

        plt.axis('tight')
        plt.show()

    @staticmethod
    def get_feature_selection():
        return feature_selection.SelectPercentile(feature_selection.f_classif)

    @staticmethod
    def get_param_headers():
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)


# class NeuralNetworkModel(ModelBase):
#
#     def get_model(self, settings):
#         batch_size = settings["batch_size"]
#         nb_epoch = settings["nb_epoch"]
#         hidden_unit_width = settings["hidden_unit_width"]
#         drop_out_rate = settings["drop_out_rate"]
#         model = Sequential()
#         model.add(Dense(input_dim=x_train.shape[1], output_dim=hidden_unit_width))
#         model.add(Activation('relu'))
#         model.add(Dropout(self.drop_out_rate))
#         model.add(Dense(input_dim=hidden_unit_width, output_dim=hidden_unit_width))
#         model.add(Activation('relu'))
#         model.add(Dropout(self.drop_out_rate))
#         model.add(Dense(input_dim=hidden_unit_width, output_dim=hidden_unit_width))
#         model.add(Activation('relu'))
#         model.add(Dropout(self.drop_out_rate))
#         model.add(Dense(output_dim=2))
#         model.add(Activation('softmax'))
#         return model
#
#     @property
#     def train(self, x_train, y_train, x_cv, y_cv):
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


class AdaBoostModel(ModelBase):

    def get_model(self, settings):
        return AdaBoostClassifier(n_estimators=settings["n_estimators"])

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def get_param_headers():
        return ["n_estimators"]


class RandomForestModel(ModelBase):

    def get_model(self, settings):
        return RandomForestClassifier(n_estimators=settings["n_estimators"],
                                      min_samples_split=settings["min_samples_split"],
                                      min_samples_leaf=settings["min_samples_leaf"],
                                      class_weight=settings["class_weight"])

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def get_param_headers():
        return ["n_estimators", "min_samples_split", "min_samples_leaf", "class_weight"]


class GaussianNBModel(ModelBase):

    def get_model(self, settings):
        return GaussianNB()

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def get_param_headers():
        return []


class SVMModel(ModelBase):

    def get_model(self, settings):
        return svm.SVC(kernel="rbf",
                       C=settings["C"],
                       class_weight=settings["class_weight"],
                       gamma=settings["gamma"])

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def get_param_headers():
        return ["C", "max_iter", "class_weight", "gamma"]



MODEL_DICT = { #"NN": NeuralNetworkModel,
               "ADA": AdaBoostModel,
               "GB": GaussianNBModel,
               "RF": RandomForestModel,
               "SVM": SVMModel}


def get_instance(model_name, settings):
    if model_name not in MODEL_DICT:
        raise ValueError("%s is not a valid options in %s" % (model_name, str(MODEL_DICT.keys())))
    else:
        model_class = MODEL_DICT[model_name]
        return model_class(settings)
