# modules
import numpy as np

from task_2_clean import train_head
from task_2_clean import test_head

########################### bootstrapping #################################

# function to perform bootstrapping on a dataframe
def bootstrap(data, n_bootstrap_samples):
    bootstrap_samples = []
    n_rows = len(data)

    for _ in range(n_bootstrap_samples):
        # Sample rows with replacement
        bootstrap_sample = data.sample(n=n_rows, replace=True)
        bootstrap_samples.append(bootstrap_sample)

    return bootstrap_samples

# number of bootstrap samples
n_bootstrap_samples = 10

# perform bootstrapping
bootstrap_samples = bootstrap(train_head, n_bootstrap_samples)

# print the first few rows of the first bootstrap sample as an example
#print(bootstrap_samples[0].head())
# compare to the original df
#print(train_head)

#############data preparation of bootstrapped samples###############################

# test data is sorted into coordinates and labels (not bootstrapped)
test = test_head.iloc[:, 0:180]
test = test * 10 ** 10
test = test.astype(int)
    # labels in separate array
test_label = test_head.iloc[:, -1]
test_label = test_label.astype(str)
test_label = np.array(test_label)
test_label = np.ravel(test_label)


# separate lists for coordinates and labels
bootstrapped_train = {}
bootstrapped_train_labels ={}

# a loop that puts coordinates and labels in respective list
for i in range(len(bootstrap_samples)):
    # training data
    boot_gesture = bootstrap_samples[i].iloc[:, 0:180]
    boot_gesture = boot_gesture * 10 ** 10
    boot_gesture = boot_gesture.astype(int)
    bootstrapped_train[i] = boot_gesture

        # training data labels
    boot_gesture_label = bootstrap_samples[i].iloc[:, -1]
    boot_gesture_label = boot_gesture_label.astype(str)
    boot_gesture_label = np.array(boot_gesture_label)
    boot_gesture_label = np.ravel(boot_gesture_label)
    bootstrapped_train_labels[i] = boot_gesture_label

#print(bootstrap_samples[1])
#print(bootstrapped_train[1])
#print(bootstrapped_train_labels[1])

