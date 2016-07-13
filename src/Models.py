__author__ = 'Rays'
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD

import timeit

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, feature_selection, cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


class ModelBase:

    def __init__(self):
        self.model = None
        pass

    def init_model(self, settings):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)

    def get_settings(self):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)

    def train(self, x_train, y_train, x_cv, y_cv):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)

    def predict(self, x):
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)

    def update_model(self, settings):
        self.model = self.init_model(settings)

    def plot_feature_selection_diagram(self, x, y):
        for index, setting in enumerate(self.get_settings()):
            self.update_model(setting)
            clf = Pipeline([('selector', self.get_feature_selection()), ('learning', self.model)])
            score_means = list()
            score_stds = list()
            percentiles = (10, 25, 40, 55, 70, 85)
            for percentile in percentiles:
                clf.set_params(selector__percentile=percentile)
                # Compute cross-validation score using all CPUs
                this_scores = cross_validation.cross_val_score(clf, x, y, n_jobs=1)
                score_means.append(this_scores.mean())
                score_stds.append(this_scores.std())
                print("Finished percentile %d" % percentile)
            plt.errorbar(percentiles, score_means, np.array(score_stds), label="Setting #{}".format(index))
        plt.title('Performance of the SVM-Anova varying the percentile of features selected')
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')
        plt.axis('tight')
        plt.legend()
        plt.show()

    def run(self, x_train, x_cv, y_train, y_cv):
        stats_list = []
        h_train = None
        h_cv = None
        for setting in self.get_settings():
            self.update_model(setting)
            start_time = timeit.default_timer()
            self.train(x_train, y_train, x_cv, y_cv)
            total_time = timeit.default_timer() - start_time
            print("Training finished with ", setting, " in ", total_time, " sec")
            # record settings
            setting["run_time"] = total_time
            train_stats, h_train = self.validate_prediction(x_train, y_train, setting, "training")
            stats_list.append(train_stats)
            cv_stats, h_cv = self.validate_prediction(x_cv, y_cv, setting, "cv")
            stats_list.append(cv_stats)
        return stats_list, h_train, h_cv

    def validate_prediction(self, x_data, y_data, settings, traing_type):
        h_train = self.model.predict(x_data)
        train_stats = self.calculate_metrics(y_data, h_train)
        train_stats["type"] = traing_type
        train_stats.update(settings)
        print("%s stats: " % traing_type)
        print("Accuracy = %3.2f%%" % (train_stats["accuracy"]))
        print("%2.2f%% bad loans predicted correctly" % (train_stats["false_accuracy"]))
        return train_stats, h_train

    def get_param_headers(self):
        headers = list(["model", "type", "accuracy", "false_accuracy", "tp", "tn", "fp", "fn",
                        "precision", "recall", "run_time"])
        headers.extend(self.get_model_param_headers())
        return headers

    @staticmethod
    def get_model_param_headers():
        raise NotImplementedError("Class %s doesn't implement aMethod()" % __name__)

    @staticmethod
    def calculate_metrics(y, h):
        size = len(y)
        error_count = 0.0
        count_dict = {}
        for i in range(size):
            if tuple((y[i], h[i])) not in count_dict.keys():
                count_dict[tuple((y[i], h[i]))] = 0
            count_dict[tuple((y[i], h[i]))] += 1
            error_count += 0 if y[i] == h[i] else 1
        accuracy = float(size - error_count) / size * 100
        true_pos = count_dict.get(tuple((1.0, 1.0)), 0)
        true_neg = count_dict.get(tuple((0.0, 0.0)), 0)
        false_pos = count_dict.get(tuple((0.0, 1.0)), 0)
        false_neg = count_dict.get(tuple((1.0, 0.0)), 0)
        return {"tp": true_pos,
                "tn": true_neg,
                "fp": false_pos,
                "fn": false_neg,
                "accuracy": accuracy,
                "false_accuracy": 0 if true_neg == 0 else true_neg*100.0/(true_neg + false_pos),
                "precision": 0 if true_pos + false_pos == 0 else true_pos * 100.0 / (true_pos + false_pos),
                "recall": 0 if true_pos + false_neg == 0 else true_pos * 100.0 / (true_pos + false_neg)}

    @staticmethod
    def get_feature_selection():
        return feature_selection.SelectPercentile(feature_selection.f_classif)

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

    def init_model(self, settings):
        return AdaBoostClassifier(n_estimators=settings["n_estimators"])

    def get_settings(self):
        return [{"n_estimators": 10},
                {"n_estimators": 20}]

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def get_model_param_headers():
        return ["n_estimators"]


class GaussianNBModel(ModelBase):

    def init_model(self, settings):
        return GaussianNB()

    def get_settings(self):
        return [{}]

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def get_model_param_headers():
        return []


class RandomForestModel(ModelBase):

    def init_model(self, settings):
        return RandomForestClassifier(n_estimators=settings["n_estimators"],
                                      min_samples_split=settings["min_samples_split"],
                                      min_samples_leaf=settings["min_samples_leaf"],
                                      class_weight=settings["class_weight"])

    def get_settings(self):
        return [{"n_estimators": n_est, "min_samples_split": min_ss, "min_samples_leaf": min_sl, "class_weight": cw}
                for n_est in [10, 30, 50]
                for min_ss in [5, 20]
                for min_sl in [2, 10]
                for cw in [{0: 0.7, 1: 0.3}, {0: 0.88, 1: 0.12}]]

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def get_model_param_headers():
        return ["n_estimators", "min_samples_split", "min_samples_leaf", "class_weight"]


class SVMModel(ModelBase):

    def init_model(self, settings):
        return svm.SVC(kernel="rbf",
                       C=settings["C"],
                       max_iter=settings["max_iter"],
                       class_weight=settings["class_weight"],
                       gamma=settings["gamma"])

    def get_settings(self):
        return [{"C": 1, "max_iter": -1, "class_weight": {0: 0.88, 1: 0.12}, "gamma": 1},
                {"C": 1, "max_iter": -1, "class_weight": {0: 0.5, 1: 0.5}, "gamma": 1}]

    def train(self, x_train, y_train, x_cv, y_cv):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def get_model_param_headers():
        return ["C", "max_iter", "class_weight", "gamma"]


MODEL_DICT = {"ADA": AdaBoostModel,
              "GB": GaussianNBModel,
              "RF": RandomForestModel,
              "SVM": SVMModel}
#"NN": NeuralNetworkModel


def get_instance(model_name):
    if model_name not in MODEL_DICT:
        raise ValueError("%s is not a valid options in %s" % (model_name, str(MODEL_DICT.keys())))
    else:
        return MODEL_DICT[model_name]()
