import csv
import numpy as np


def separate_training_data_file(fn, training_ratio, cv_ratio, random_seeds):
    training = []
    cv = []
    test = []
    seed_len = len(random_seeds)
    i = 0
    with open(fn) as f:
        for line in f:
            if line.startswith("\"id\""):
                if len(training) == 0:
                    training.append(line)
                    cv.append(line)
                    test.append(line)
            elif line.startswith("\""):
                random_seed = random_seeds[i % seed_len]
                if random_seed < training_ratio:
                    training.append(line)
                elif random_seed < training_ratio+cv_ratio:
                    cv.append(line)
                else:
                    test.append(line)
                i += 1
    write_to_file(fn, "training", training)
    write_to_file(fn, "cv", cv)
    write_to_file(fn, "test", test)


def write_to_file(fn, postfix, lines):
    sample_file = fn.replace(".csv", "_"+postfix+".csv")
    f = open(sample_file, "w")
    f.writelines(lines)
    f.close()
    print("Wrote to %s with lines(including header): %d" % (sample_file, len(lines)))


# def load_mapped_feature(fn):
#     data = np.load(fn)
#     y = data['y']
#     x = data['x']
#     return x, y
#
#
# def save_mapped_feature(x, y, fn):
#     np.savez(fn, x=x, y=y)


def load_random_seeds(fn="data/random_seeds.npy"):
    return np.load(fn)


def save_random_seeds(random_seeds, fn="data/random_seeds"):
    np.save(fn, np.asarray(random_seeds))


def save_results(headers, stats_list, fn):
    with open(fn, "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(headers)
        for stats in stats_list:
            row = []
            for header in headers:
                row.append(stats.get(header))
            writer.writerow(row)
