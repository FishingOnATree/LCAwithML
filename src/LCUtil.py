import numpy as np
import random

def separate_training_data(x, y, trainging_ratio, cv_ratio):
    print("randomly generated random seeds")
    return separate_training_data(x, y, trainging_ratio, cv_ratio, [random.random() for _ in range(x.shape[0])])


def separate_training_data(x, y, trainging_ratio, cv_ratio, random_seeds):
    m, n = x.shape
    x_train = []
    x_cv = []
    x_test = []
    y_train = []
    y_cv = []
    y_test = []
    seed_len = len(random_seeds)
    for i in range(m):
        random_seed = random_seeds[i % seed_len]
        if random_seed < trainging_ratio:
            x_train.append(x[i])
            y_train.append(y[i])
        elif random_seed < trainging_ratio+cv_ratio:
            x_cv.append(x[i])
            y_cv.append(y[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_cv = np.array(x_cv)
    y_cv = np.array(y_cv)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, x_cv, x_test, y_train.ravel(), y_cv.ravel(), y_test.ravel()


def cal_accuracy(y, h):
    m = len(y)
    error_count = 0.0
    count_dict = {}
    for i in range(m):
        if tuple((y[i], h[i])) not in count_dict.keys():
            count_dict[tuple((y[i], h[i]))] = 0
        count_dict[tuple((y[i], h[i]))] += 1
        error_count += 0 if y[i] == h[i] else 1
    return error_count, m, count_dict


def load_mapped_feature(fn):
    data = np.load(fn)
    y = data['y']
    x = data['x']
    return x, y


def save_mapped_feature(x, y, fn):
    np.savez(fn, x=x, y=y)


def load_random_seeds(fn="data/random_seeds.npy"):
    return np.load(fn)


def save_random_seeds(random_seeds, fn="data/random_seeds"):
    np.save(fn, np.asarray(random_seeds))