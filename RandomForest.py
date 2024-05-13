# Data Processing
import pandas as pd
import numpy as np
from main import test_processed
from main import train_processed

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# Data


def dataprep(train):
    y = train['index']
    x = train.drop(['index', 'word'], axis=1)
    print(f'x : {x.shape}')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)
    print(f'x_train : {x_train.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'x_test : {x_test.shape}')
    print(f'y_test : {y_test.shape}')
    return x_train, x_test, y_train, y_test


'''
def random_forest(train, test):
    rf = RandomForestClassifier()
    rf.fit(train)
    pred = rf.predict(test)
    accuracy = accuracy_score(test, pred)
    print(f'Accuracy: {accuracy}')

    test_pred = rf.predict(test)


random_forest(train_blind, test_blind)
'''


def random_forest_2(x, y):
    x_train, x_test, y_train, y_test = dataprep(x)

    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=20)
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


# TA BORT KOORDINATERNA

# random_forest_2(train_processed, test_processed)

# def random_forest_visualization():
