from __future__ import print_function
import datetime
import numpy as np
import random
import time
import timeit
import sys

from sklearn import preprocessing
from sklearn.externals import joblib

import LCCfg
import LCUtil
import FeatureMapping
import Models


def data_preprocess(df_train, df_cv, df_test):
    x_train,  y_train = FeatureMapping.map_features(df_train)
    x_cv, y_cv = FeatureMapping.map_features(df_cv)
    x_test, y_test = FeatureMapping.map_features(df_test)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_cv = scaler.transform(x_cv)
    x_test = scaler.transform(x_test)
    return x_train, x_cv, x_test, y_train, y_cv, y_test


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
    return train_stats, h_train


def map_random(x):
    return random.random()


def name_file(settings):
    return "weights_C" + str(settings["C"]) + "_max_iter" \
           + str(settings["max_iter"]) \
           + "_NW" + str(settings["class_weight"][0]) \
           + "_" + get_time_str() \
           + ".pkl"


def get_time_str():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M')


def get_seperated_data_frame(data_f):
    # load data
    df = FeatureMapping.load_data(data_f)
    df["random"] = df["id"].map(map_random)
    df_train = df[df["random"] < 0.6]
    df_cv = df[(df["random"] < 0.8) & (df["random"] >= 0.6)]
    df_test = df[df["random"] >= 0.8]
    return df_train, df_cv, df_test


def get_separated_data_and_frame(data_f):
    df_train, df_cv, df_test = get_seperated_data_frame(data_f)
    print(df_train.shape[0], df_cv.shape[0], df_test.shape[0])
    # need to normalize mean and standardization
    x_train, x_cv, x_test, y_train, y_cv, y_test = data_preprocess(df_train, df_cv, df_test)
    return df_train, df_cv, df_test,  x_train,x_cv, x_test, y_train, y_cv, y_test


def train(data_f, model_name):
    df_train, df_cv, df_test,  x_train,x_cv, x_test, y_train, y_cv, y_test = get_separated_data_and_frame(data_f)

    print("Data size: Training, CV, Test = %d, %d, %d" % (x_train.shape[0], x_cv.shape[0], x_test.shape[0]))
    stats_list = []
    model = None
    for iteration in [[1, {0: 0.88, 1: 0.12}]]:
                c = iteration[0]
                class_weights = iteration[1]
                max_iter = -1
                # TODO: figure out a better way to specify different parameters for each model.
                settings = {"C": c,
                            "max_iter": max_iter,
                            "cache_size": 10000,
                            "class_weight": class_weights,
                            "gamma": 1,
                            "min_samples_split": 12,
                            "min_samples_leaf": 5,
                            "n_estimators": 50,
                            "model": model_name}
                model = Models.get_instance(model_name, settings)
                start_time = timeit.default_timer()
                model.train(x_train, y_train, x_cv, y_cv)
                total_time = timeit.default_timer() - start_time
                print("Training finished with ", settings, " in ", total_time, " sec")
                # record settings
                settings["run_time"] = total_time
                train_stats, h_train = validate_prediction(model, x_train, y_train, settings, "training")
                # df_train["prediction"] = np.copy(h_train)
                stats_list.append(train_stats)
                cv_stats, h_cv = validate_prediction(model, x_cv, y_cv, settings, "cv")
                # df_cv["prediction"] = np.copy(h_cv)
                stats_list.append(cv_stats)
                test_stats, h_test = validate_prediction(model, x_test, y_test, settings, "test")
                df_test["prediction"] = np.copy(h_test)
                stats_list.append(test_stats)

                # df_test.to_csv(config.data_dir+"/df_test.csv")
                # # save final weight
                # weight_f = config.data_dir + "/weights/" + name_file(settings)
                # joblib.dump(model, weight_f)
    headers = ["model", "type", "accuracy", "false_accuracy", "tp", "tn", "fp", "fn", "run_time"]
    headers.extend(model.get_param_headers())

    time_str = get_time_str()
    out_put_fn = config.data_dir + "/" + model_name + "_" + time_str + ".csv"
    LCUtil.save_results(headers, stats_list, out_put_fn)
    return df_train, df_cv, df_test, model


def predict(data_f, model):
    # load data
    df = FeatureMapping.load_data(data_f)
    x, y = FeatureMapping.map_features(df)
    print(x.shape)
    # need to normalize mean and standardization
    scaler = preprocessing.StandardScaler().fit(x)
    x_norm = scaler.transform(x)
    h = model.predict(x_norm)
    df["prediction"] = np.copy(h)
    return df


def feature_selection(data_f, model_name):
    df_train, df_cv, df_test,  x_train,x_cv, x_test, y_train, y_cv, y_test = get_separated_data_and_frame(data_f)
    for iteration in [[1, {0: 0.75, 1: 0.25}]]:
        c = iteration[0]
        class_weights = iteration[1]
        max_iter = -1
        settings = {"C": c,
                    "max_iter": max_iter,
                    "cache_size": 10000,
                    "class_weight": class_weights,
                    "gamma": 1,
                    "min_samples_split": 12,
                    "min_samples_leaf": 5,
                    "n_estimators": 50,
                    "model": model_name}
        model = Models.get_instance(model_name, settings)
        model.plot_feature_selection_diagram(x_train, y_train)


def main():
    if len(sys.argv) < 4:
        print('not enough arguments')
        sys.exit()
    else:
        model_name = sys.argv[2]
        # use sample data by default
        data_file = config.data_dir + "/" + sys.argv[3]

        if sys.argv[1] == "train":
            train(data_file, model_name)
        elif sys.argv[1] == "feature_select":
            feature_selection(data_file, model_name)
        else:
            if len(sys.argv) < 5:
                print('not enough arguments')
                sys.exit()
            else:
                weight_file = sys.argv[4]
                model = joblib.load(weight_file)
                df = predict(data_file, model)
                df.to_csv(config.data_dir+"/prediction_result_" + get_time_str() + ".csv")


config = LCCfg.LCCfg("default.cfg")
if __name__ == "__main__":
    main()
