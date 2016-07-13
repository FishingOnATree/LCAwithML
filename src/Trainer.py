from __future__ import print_function
import datetime
import random
import time
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
    return df_train, df_cv, df_test,  x_train, x_cv, x_test, y_train, y_cv, y_test


def train(data_f, model, model_name):
    df_train, df_cv, df_test,  x_train, x_cv, x_test, y_train, y_cv, y_test = get_separated_data_and_frame(data_f)
    print("Data size: Training, CV, Test = %s, %s, %s" % (str(x_train.shape), str(x_cv.shape), str(x_test.shape)))
    print("Label size: Training, CV, Test = %s, %s, %s" % (str(y_train.shape), str(y_cv.shape), str(y_test.shape)))
    stats_list, h_train, h_cv = model.run(x_train, x_cv, y_train, y_cv)
    time_str = get_time_str()
    out_put_fn = config.data_dir + "/" + model_name + "_" + time_str + ".csv"
    print(model.get_param_headers())
    LCUtil.save_results(model.get_param_headers(), stats_list, out_put_fn)
    # df_test.to_csv(config.data_dir+"/df_test.csv")
    # # save final weight
    # weight_f = config.data_dir + "/weights/" + name_file(settings)
    # joblib.dump(model, weight_f)
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
    # TODO: need a better way to copy: df["prediction"] = np.copy(h)
    return df


def feature_selection(data_f, model):
    df_train, df_cv, df_test,  x_train,x_cv, x_test, y_train, y_cv, y_test = get_separated_data_and_frame(data_f)
    model.plot_feature_selection_diagram(x_train, y_train)


def main():
    if len(sys.argv) < 4:
        print('not enough arguments')
        sys.exit()
    else:
        purpose = sys.argv[1]
        model_name = sys.argv[2]
        # use sample data by default
        data_file = config.data_dir + "/" + sys.argv[3]

        if purpose == "train":
            model = Models.get_instance(model_name)
            train(data_file, model, model_name)
        elif purpose == "feature_selection":
            model = Models.get_instance(model_name)
            feature_selection(data_file, model)
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
