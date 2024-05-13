# Data Processing
import pandas as pd
import numpy as np
from main import test_processed
from main import train_processed

import bootstrap

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn.tree import export_text
import matplotlib.pyplot as plt

# Data


def dataprep(train_df, test_df):

    train_features = train_df.iloc[:, :-1].values.tolist()
    train_target = train_df.iloc[:, -1].values.tolist()
    train = (train_features, train_target)

    test_features = test_df.iloc[:, :-1].values.tolist()
    test_target = test_df.iloc[:, -1].values.tolist()
    test = (test_features, test_target)

    return train, test


def random_forest_2(train, test):
    train, train_target = bootstrap.knn_prep(train)
    test, test_target = bootstrap.knn_prep(test)

    rf_model = RandomForestClassifier()
    rf_model.fit(train, train_target)

    pred = rf_model.predict(test)

    accuracy = accuracy_score(pred, test_target)
    print("Accuracy:", accuracy)

    # Need better plot with graph
    for i in range(3):
        tree = rf_model.estimators_[i]
        tree_rules = export_text(tree, feature_names=train.columns.tolist(), max_depth=2)
        print(f"Tree {i + 1}:\n{tree_rules}")

        # Plots the tree structure
        plt.figure(figsize=(10, 6))
        tree.plot_tree(tree, feature_names=train.columns, filled=True, max_depth=2, proportion=True)
        plt.show()


random_forest_2(train_processed, test_processed)


'''
def random_forest_visualization():
    for i in range(3):
        tree = rf.estimators_[i]
        dot_data = export_graphviz(tree,
                                   feature_names=X_train.columns,
                                   filled=True,
                                   max_depth=2,
                                   impurity=False,
                                   proportion=True)
        graph = graphviz.Source(dot_data)
        display(graph)
'''