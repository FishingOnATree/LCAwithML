from __future__ import print_function
import datetime
import random
import time
import timeit
import sys

from sklearn import preprocessing

import LCCfg
import LCUtil
import FeatureMapping
from models import SVMModel #, NeuralNetworkModel
from sklearn.externals import joblib


def data_preprocess(x, y):
    # use pre-saved random seeds to ensure the same train/cv/test set
    random_seeds = LCUtil.load_random_seeds()
    xtrain, xcv, xtest, ytrain, ycv, ytest = \
        LCUtil.separate_training_data(x, y, 0.6, 0.2, random_seeds)
    scaler = preprocessing.StandardScaler().fit(xtrain)
    xtrain = scaler.transform(xtrain)
    xcv = scaler.transform(xcv)
    xtest = scaler.transform(xtest)
    return xtrain, xcv, xtest, ytrain, ycv, ytest


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


def validate_prediction(model, x_data, y_data, settings, traing_type):
    h_train = model.predict(x_data)
    train_stats = cal_accuracy(y_data, h_train)
    train_stats["type"] = traing_type
    train_stats.update(settings)
    print("%s stats: " % traing_type)
    print("Accuracy = %3.2f%%" % (train_stats["accuracy"]))
    print("%2.2f%% bad loans predicted correctly" % (train_stats["false_accuracy"]))
    return train_stats


def map_random(x):
    return random.random()


def train(data_f, weight_f):
    # load data
    df = FeatureMapping.load_data(data_f)
    df["random"] = df["id"].map(map_random)
    # map polynomial features
    poly_degree = 1
    # poly = preprocessing.PolynomialFeatures(degree=poly_degree, interaction_only=True)
    # x = poly.fit_transform(x)
    df_train = df[df["random"] < 0.6]
    df_cv = df[(df["random"] < 0.8) & (df["random"] >= 0.6)]
    df_test = df[df["random"] >= 0.8]

    print(df_train.shape[0], df_cv.shape[0], df_test.shape[0])
    # need to normalize mean and standardization
    x_train,  y_train = FeatureMapping.map_features(df_train)
    x_cv, y_cv = FeatureMapping.map_features(df_cv)
    x_test, y_test = FeatureMapping.map_features(df_test)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_cv = scaler.transform(x_cv)
    x_test = scaler.transform(x_test)

    print("Data size: Training, CV, Test = %d, %d, %d" % (x_train.shape[0], x_cv.shape[0], x_test.shape[0]))
    stats_list = []
    print("SVM trainer")
    model = None
    for c in [1, 10, 100, 1000]:
        for max_iter in [-1]: #range(6000, 17001, 1000):
            for neg_weights in range(80, 95, 3):
                neg_weight_float = neg_weights / 100.0
                class_weights = {0: neg_weight_float, 1: (1-neg_weight_float)}
                settings = {"C": c,
                            "max_iter": max_iter,
                            "cache_size": 10000,
                            "class_weight": class_weights}
                model = SVMModel.SVMModel(settings)
                start_time = timeit.default_timer()
                model.train(x_train, y_train, x_cv, y_cv)
                total_time = timeit.default_timer() - start_time
                print("Training finished with ", settings, " in ", total_time, " sec")
                # record settings
                settings["run_time"] = total_time
                settings["poly_degree"] = poly_degree
                # train_stats = validate_prediction(model, x_train, y_train, settings, "training")
                # stats_list.append(train_stats)
                cv_stats = validate_prediction(model, x_cv, y_cv, settings, "cv")
                stats_list.append(cv_stats)
    headers = ["type", "poly_degree", "C", "max_iter", "cache_size", "class_weight",
               "accuracy", "false_accuracy", "tp", "tn", "fp", "fn", "run_time"]

    time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M')
    out_put_fn = config.data_dir + "/" + time_str + ".csv"
    # LCUtil.save_results(headers, stats_list, out_put_fn)
    # # save final weight
    # joblib.dump(model, weight_f)


def predict(data_file, weight_file):
    model = joblib.load(weight_file)
    # load data
    df = FeatureMapping.load_data(data_file)
    x, y = FeatureMapping.map_features(df)
    # map polynomial features
    poly_degree = 1
    poly = preprocessing.PolynomialFeatures(degree=poly_degree, interaction_only=True)
    x = poly.fit_transform(x)
    print(x.shape)
    # need to normalize mean and standardization
    x_train, x_cv, x_test, y_train, y_cv, y_test = data_preprocess(x, y)
    stats = validate_prediction(model, x_test, y_test, {}, "test")
    print(stats)


if len(sys.argv) < 3:
    print('not enough arguments')
    sys.exit()
else:
    config = LCCfg.LCCfg("default.cfg")
    is_trainging = True if sys.argv[1] == "train" else False
    # use sample data by default
    data_file = config.data_dir + "/" + sys.argv[2]
    weight_file = config.data_dir + "/" + sys.argv[3]

    if is_trainging:
        train(data_file, weight_file)
    else:
        predict(data_file, weight_file)


