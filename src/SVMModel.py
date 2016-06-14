__author__ = 'Rays'

import LCUtil
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
    print("%2.2f%% bad loans predicted correctly" % (true_neg*100/float(true_neg+false_pos)))


def train(x_train, y_train, x_cv, y_cv):
    clf = svm.SVC(C=10000, max_iter=150000, cache_size=3000, class_weight={0:0.8, 1:0.2})
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
