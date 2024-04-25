# KNN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('train-final.csv', encoding='latin-1', sep=',')
test = pd.read_csv('test-final.csv', encoding='latin-1', sep=',')
