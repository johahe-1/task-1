# modules
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib as plt  # Visualisering
import matplotlib.pyplot as pltplot  # Visualisering
import bootstrap


def rknn(train_processed, test_processed, maxrange):
    # a loop that tests a range of k values and chooses optimal by accuracy score

    # empty list for storing mean_squared_error score
    errors = []
    accuracies = []
    best_accuracy = 0
    best_k = 0

    # all k
    ks = [x for x in range(1, maxrange) if x % 2 != 0]

    # data
    bs_train, bs_train_label = bootstrap.bootstrap_dict(train_processed, 10)
    test, test_label = bootstrap.knn_prep(test_processed)

    for k in ks:
        for i in bs_train.keys():
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(bs_train[i], bs_train_label[i])
        label_pred = knn.predict(test)
        error = mean_squared_error(test_label, label_pred)
        errors.append(error)
        accuracy = accuracy_score(test_label, label_pred)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
    print('best accuracy for k = (1,', maxrange, '):', best_accuracy)
    print('best k:', best_k)
    knn_plot(ks, errors)


def knn_plot(ks, errors):
    pltplot.plot(ks, errors)
    pltplot.xlabel('k')
    pltplot.ylabel('Error')
    pltplot.title('Error Curve for knn Algorithm (Bootstrapped training data)')
    pltplot.show()
