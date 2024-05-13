# modules
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib as plt # Visualisering
import matplotlib.pyplot as plt # Visualisering

# data and labels
    # bootstrapped training data split into dataframes and raveled array for labels (in dictionaries)
from bootstrap import bootstrapped_train
from bootstrap import bootstrapped_train_labels

    # testing data split into dataframe and raveled array for labels
from bootstrap import test
from bootstrap import test_label

########################## knn optimization ######################

# testing different values of k neighbors
max = 500  # max value for the range
ks = [x for x in range(1, max) if x % 2 != 0]  # choose a range with uneven integers

# empty list for storing mean_squared_error score
errors = []
accuracies = []
best_accuracy = 0
best_k = 0


# a loop that tests a range of k values and chooses optimal by accuracy score
def funktionsnamn(max,ks):
    for k in ks:
        for i in bootstrapped_train.keys():
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(bootstrapped_train[i], bootstrapped_train_labels[i])
        label_pred = knn.predict(test)
        error = mean_squared_error(test_label, label_pred)
        errors.append(error)
        accuracy = accuracy_score(test_label, label_pred)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
    print('best accuracy for k = (1,',max,'):', best_accuracy)
    print('best k:', best_k)


# FIXA FUNKTION
def funktionsnamn(x,y):
    for k in ks:
        for i in x.keys():
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x[i], y[i])
        label_pred = knn.predict(test)
        error = mean_squared_error(test_label, label_pred)
        errors.append(error)
        accuracy = accuracy_score(test_label, label_pred)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
    print('best accuracy for k = (1,',max,'):', best_accuracy)
    print('best k:', best_k)

#funktionsnamn(bootstrapped_train[i], bootstrapped_train_labels[i])


# plot the error curve for different k's
plt.plot(ks, errors)
plt.xlabel('k')
plt.ylabel('Error')
plt.title('Error Curve for knn Algorithm (Bootstrapped training data)')
plt.show()
