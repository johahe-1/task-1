# modules
import numpy as np


def bootstrap(data, n):
    # function to perform bootstrapping on a dataframe
    bootstrap_samples = []
    n_rows = len(data)

    for _ in range(n):
        # Sample rows with replacement
        bootstrap_sample = data.sample(n=n_rows, replace=True)
        bootstrap_samples.append(bootstrap_sample)

    return bootstrap_samples


def knn_prep(data):
    # test data is sorted into coordinates and labels (not bootstrapped)
    test = data.iloc[:, 0:180]
    test = test * 10 ** 8  # obs skär bort en decimal
    test = test.astype(int)

    # labels in separate array
    test_label = data.iloc[:, -1]
    test_label = test_label.astype(str)
    test_label = np.array(test_label)
    test_label = np.ravel(test_label)

    return test, test_label


def bootstrap_dict(data, n):
    bootstrap_samples = bootstrap(data, n)
    # separate lists for coordinates and labels
    bootstrapped_train = {}
    bootstrapped_train_labels = {}

    # a loop that puts coordinates and labels in respective list
    for i in range(len(bootstrap_samples)):
        # training data
        boot_gesture = bootstrap_samples[i].iloc[:, 0:180]
        boot_gesture = boot_gesture * (10 ** 8)  # obs, skär bort 1 decimal
        boot_gesture = boot_gesture.astype(int)
        bootstrapped_train[i] = boot_gesture

        # training data labels
        boot_gesture_label = bootstrap_samples[i].iloc[:, -1]
        boot_gesture_label = boot_gesture_label.astype(str)
        boot_gesture_label = np.array(boot_gesture_label)
        boot_gesture_label = np.ravel(boot_gesture_label)
        bootstrapped_train_labels[i] = boot_gesture_label

    return bootstrapped_train, bootstrapped_train_labels
