# resources
from task_2_clean import train_head
from task_2_clean import test_head

# modules
import pandas as pd
import numpy as np

#################### data preparation ####################################
# complete datasets split into coordinates and labels for each gesture

# training data
train = train_head.iloc[:, 0:180]
train = train * 10 ** 10
train = train.astype(int)
    # labels in separate array
train_label = train_head.iloc[:, -1]
train_label = train_label.astype(str)
train_label = np.array(train_label)
train_label = np.ravel(train_label)


# test data is also sorted
test = test_head.iloc[:, 0:180]
test = test * 10 ** 10
test = test.astype(int)
    # labels in separate array
test_label = test_head.iloc[:, -1]
test_label = test_label.astype(str)
test_label = np.array(test_label)
test_label = np.ravel(test_label)