from __future__ import print_function
import LCCfg
import LCUtil
import NNTrain
import SVMTrainer
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

        # use pre-saved random seeds to ensure the same train/cv/test set
        random_seeds = LCUtil.load_random_seeds()

        # need to normalize mean and standardization
        x_train, x_cv, x_test, y_train, y_cv, y_test = \
            LCUtil.separate_training_data(x, y, 0.6, 0.2, random_seeds)

        x_train = preprocessing.scale(x_train)
        x_cv = preprocessing.scale(x_cv)
        x_test = preprocessing.scale(x_test)

        print("Data size: Training, CV, Test = %d, %d, %d" % (x_train.shape[0], x_cv.shape[0], x_test.shape[0]))
        print(x_train[0, :])
        option = sys.argv[2]
        if option.startswith("nn"):
            print("NN trainer")
            NNTrain.train(x_train, y_train, x_cv, y_cv)
        elif option.startswith("svm"):
            print("SVM trainer")
            SVMTrainer.train(x_train, y_train, x_cv, y_cv)