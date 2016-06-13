__author__ = 'Rays'

import LCUtil
from sklearn import preprocessing
from sklearn import svm


def print_results(error, size, cond_dict):
    print("Accuracy = %3.2f%%" % ((size - error) / size * 100))
    true_pos = cond_dict[tuple((1.0, 1.0))] if tuple((1.0, 1.0)) in cond_dict.keys() else 0
    true_neg = cond_dict[tuple((0.0, 0.0))] if tuple((0.0, 0.0)) in cond_dict.keys() else 0
    false_pos = cond_dict[tuple((0.0, 1.0))] if tuple((0.0, 1.0)) in cond_dict.keys() else 0
    false_neg = cond_dict[tuple((1.0, 0.0))] if tuple((1.0, 0.0)) in cond_dict.keys() else 0
    print(true_pos, true_neg, false_pos, false_neg)
    print("Precision = %.4f" % (true_pos/float(true_pos+false_pos)))
    print("Recall = %.4f" % (true_pos/float(true_pos+false_neg)))


x, y = LCUtil.load_mapped_feature()

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


clf = svm.SVC(C=10000, max_iter=-1, cache_size=1000, class_weight={0:0.8, 1:0.2})
clf.fit(x_train, y_train)

h_train = clf.predict(x_train)
error_train, size_train, cond_dict_train = LCUtil.cal_accuracy(y_train, h_train)
print("Training stats: ")
print_results(error_train, size_train, cond_dict_train)
print(cond_dict_train)

h_cv = clf.predict(x_cv)
error_cv, size_cv, cond_dict1_cv = LCUtil.cal_accuracy(y_cv, h_cv)
print("CV stats: ")
print_results(error_cv, size_cv, cond_dict1_cv)
print(cond_dict1_cv)
