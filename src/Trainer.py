from __future__ import print_function
import sys
import datetime
import time
import timeit

from sklearn import preprocessing

import LCCfg
import LCUtil
from models import SVMModel, NeuralNetworkModel


def data_preprocess(x, y):
    # use pre-saved random seeds to ensure the same train/cv/test set
    random_seeds = LCUtil.load_random_seeds()

    xtrain, xcv, xtest, ytrain, ycv, ytest = \
        LCUtil.separate_training_data(x, y, 0.6, 0.2, random_seeds)
    scaler = preprocessing.StandardScaler().fit(xtrain)
    xtrain = scaler.transform(xtrain)
    xcv = scaler.transform(xcv)
    xtest = scaler.transform(xtest)
    return xtrain, xcv, xtest, ytrain, ycv, ytest, scaler


def cal_accuracy(y, h):
    size = len(y)
    error_count = 0.0
    count_dict = {}
    for i in range(size):
        if tuple((y[i], h[i])) not in count_dict.keys():
            count_dict[tuple((y[i], h[i]))] = 0
        count_dict[tuple((y[i], h[i]))] += 1
        error_count += 0 if y[i] == h[i] else 1
    accuracy = float(size - error_count) / size * 100
    true_pos = count_dict[tuple((1.0, 1.0))] if tuple((1.0, 1.0)) in count_dict.keys() else 0
    true_neg = count_dict[tuple((0.0, 0.0))] if tuple((0.0, 0.0)) in count_dict.keys() else 0
    false_pos = count_dict[tuple((0.0, 1.0))] if tuple((0.0, 1.0)) in count_dict.keys() else 0
    false_neg = count_dict[tuple((1.0, 0.0))] if tuple((1.0, 0.0)) in count_dict.keys() else 0
    bad_loan_accuracy = 0 if true_neg == 0 else true_neg*100/float(true_neg+false_pos)
    return {"tp": true_pos,
            "tn": true_neg,
            "fp": false_pos,
            "fn": false_neg,
            "accuracy": accuracy,
            "false_accuracy": bad_loan_accuracy}


def train_and_validate(model, x_train, y_train, x_cv, y_cv):
    model.train(x_train, y_train, x_cv, y_cv)
    h_train = model.predict(x_train)
    h_cv = model.predict(x_cv)
    train_stats = cal_accuracy(y_train, h_train)
    train_stats["type"] = "training"
    print("Training stats: ")
    print("Accuracy = %3.2f%%" % (train_stats["accuracy"]))
    print("%2.2f%% bad loans predicted correctly" % (train_stats["false_accuracy"]))

    cv_stats = cal_accuracy(y_cv, h_cv)
    cv_stats["type"] = "cv"
    print("CV stats: ")
    print("Accuracy = %3.2f%%" % (cv_stats["accuracy"]))
    print("%2.2f%% bad loans predicted correctly" % (cv_stats["false_accuracy"]))
    return train_stats, cv_stats


def run_training_iteration(model, x_train, y_train, x_cv, y_cv):
    start_time = timeit.default_timer()
    train_stats, cv_stats = train_and_validate(model, x_train, y_train, x_cv, y_cv)
    total_time = timeit.default_timer() - start_time
    settings["run_time"] = total_time
    settings["poly_degree"] = poly_degree
    train_stats.update(settings)
    cv_stats.update(settings)
    print("Finished with ", settings, " in ", total_time, " sec")
    return train_stats, cv_stats


if len(sys.argv) < 3:
    print('not enough arguments')
    sys.exit()
else:
    config = LCCfg.LCCfg("default.cfg")
    # use sample data by default
    if sys.argv[1] == "full":
        training_data = config.data_dir + "/" + config.training_full
    else:
        training_data = config.data_dir + "/" + config.training_sample

    if sys.argv[2] in ("nn", "svm"):
        # load data
        x, y = LCUtil.load_mapped_feature(training_data + ".npz")

        # map polynomial features
        poly_degree = 2
        poly = preprocessing.PolynomialFeatures(degree=poly_degree, interaction_only=True)
        x = poly.fit_transform(x)
        print(x.shape)

        # need to normalize mean and standardization
        x_train, x_cv, x_test, y_train, y_cv, y_test, scaler = data_preprocess(x, y)

        print("Data size: Training, CV, Test = %d, %d, %d" % (x_train.shape[0], x_cv.shape[0], x_test.shape[0]))
        print(x_train[0, :])
        option = sys.argv[2]
        model = None
        settings = None
        headers = None
        stats_list = []
        if option.startswith("nn"):
            print("NN trainer")
            settings = {"batch_size": 32,
                        "nb_epoch": 10,
                        "hidden_unit_width": 300,
                        "drop_out_rate": 0.25}
            model = NeuralNetworkModel.NeuralNetworkModel(settings)
            train_stats, cv_stats = run_training_iteration(model, x_train, y_train, x_cv, y_cv)
            stats_list.append(train_stats)
            stats_list.append(cv_stats)
            headers = ["type", "poly_degree", "nb_epoch", "hidden_unit_width", "drop_out_rate",
                       "accuracy", "false_accuracy", "tp", "tn", "fp", "fn", "run_time"]
        elif option.startswith("svm"):
            print("SVM trainer")
            for c in [0.0001, 0.01, 1, 100, 10000]:
                for max_iter in range(150000, 650001, 250000):
                    for class_weights in [{0: 0.5, 1: 0.5}, {0: 0.9, 1: 0.1}, {0: 0.97, 1: 0.03}]:
                        settings = {"C": c,
                                    "max_iter": max_iter,
                                    "cache_size": 1000,
                                    "class_weight": class_weights}
                        model = SVMModel.SVMModel(settings)
                        train_stats, cv_stats = run_training_iteration(model, x_train, y_train, x_cv, y_cv)
                        stats_list.append(train_stats)
                        stats_list.append(cv_stats)
            headers = ["type", "poly_degree", "C", "max_iter", "cache_size", "class_weight",
                       "accuracy", "false_accuracy", "tp", "tn", "fp", "fn", "run_time"]

        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M')
        out_put_fn = config.data_dir + "/" + option + time_str + ".csv"
        LCUtil.save_results(headers, stats_list, out_put_fn)
