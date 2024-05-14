# modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from scipy.stats import norm

def samp_rep(data, samples, samp_rows): # data to sample, how many samples, and how many rows sampled
    # sampling with replacement on a dataframe
    rng = np.random.default_rng()

    bootstrap_samples = []
    for _ in range(samples): # just iterating the loop n times
        # Sample rows with replacement
        bootstrap_sample = data.sample(n=samp_rows, replace=True, random_state=rng)
        bootstrap_samples.append(bootstrap_sample)

    return bootstrap_samples

def boot_stats(bootstrapped_datasets):
    bootstrap_std = []

    for sample in range(len(bootstrapped_datasets)):
        standard = np.std(bootstrapped_datasets[sample]) #standard deviation
        bootstrap_std.append(standard)

    # Plot the distribution of bootstrap statistics
    fig, ax = plt.subplots()
    ax.hist(bootstrap_std, bins=20)
    ax.set_title('Bootstrap Distribution')
    ax.set_xlabel('Statistic Value')
    ax.set_ylabel('Frequency')
    plt.show()

# unintegrated bootstrap function below, cannot extract samples
def bootstrapper(data, samples):
    # function to perform bootstrapping on a dataframe
    data = data.iloc[:10, 1:240] * (10 ** 8)
    data = data.astype(int)

    dist = norm(loc=2, scale=4)
    std_true = dist.std()  # the true value of the statistic
    print('std_true:', std_true)

    std_sample = np.std(data)  # the sample statistic
    print('std_sample:', std_sample)

    rng = np.random.default_rng()
    data = (data,)  # samples must be in a sequence
    res = bootstrap(data, np.std, n_resamples=samples, confidence_level=0.9,
                    random_state=rng)

    fig, ax = plt.subplots()
    ax.hist(res.bootstrap_distribution, bins=10)
    ax.set_title('Bootstrap Distribution')
    ax.set_xlabel('statistic value')
    ax.set_ylabel('frequency')
    plt.show()

# data prep of bootstrapped data for classifiers
def knn_prep(data):
    # test data is sorted into coordinates and labels (not bootstrapped)
    test = data.iloc[:, 0:240]
    test = test * 10 ** 8  # obs skär bort en decimal
    test = test.astype(int)

    # labels in separate array
    test_label = data.iloc[:, -1]
    test_label = test_label.astype(str)
    test_label = np.array(test_label)
    test_label = np.ravel(test_label)

    return test, test_label


def bootstrap_dict(data, samples, samp_rows):
    bootstrap_samples = samp_rep(data, samples, samp_rows)
    # separate lists for coordinates and labels
    bootstrapped_train = {}
    bootstrapped_train_labels = {}

    # a loop that puts coordinates and labels in respective list
    for i in range(len(bootstrap_samples)):
        # training data
        boot_gesture = bootstrap_samples[i].iloc[:, 0:240]
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
