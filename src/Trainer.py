from __future__ import print_function
import LCCfg
import LCUtil
import NeuralNetworkModel
import SVMModel
from sklearn import preprocessing
import sys


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
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)
        x = poly.fit_transform(x)

        print(x.shape)

        # use pre-saved random seeds to ensure the same train/cv/test set
        random_seeds = LCUtil.load_random_seeds()

        # need to normalize mean and standardization
        x_train, x_cv, x_test, y_train, y_cv, y_test = \
            LCUtil.separate_training_data(x, y, 0.6, 0.2, random_seeds)
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_cv = scaler.transform(x_cv)
        x_test = scaler.transform(x_test)

        print("Data size: Training, CV, Test = %d, %d, %d" % (x_train.shape[0], x_cv.shape[0], x_test.shape[0]))
        print(x_train[0, :])
        option = sys.argv[2]
        if option.startswith("nn"):
            print("NN trainer")
            NeuralNetworkModel.train(x_train, y_train, x_cv, y_cv)
        elif option.startswith("svm"):
            print("SVM trainer")
            SVMModel.train(x_train, y_train, x_cv, y_cv)

        #TODO logging parameters, results, time. Trainer should only receive