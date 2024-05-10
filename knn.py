import

import sklearn
from sklearn.neighbors import KNeighborsClassifier

# training data
train = train_head.iloc[:, 0:60]

train_label = train_head.iloc[:, 241:242]
train_label = train_label.astype(str)
train_label = np.array(train_label)
train_label = np.ravel(train_label)

train = train * 10 ** 10
train = train.astype(int)

# test data
test = test_head.iloc[:, 0:60]
test_label = test_head.iloc[:, 241:242]
test_label = test_label.astype(str)
test_label = np.array(test_label)
test_label = np.ravel(test_label)
test = test * 10 ** 10
test = test.astype(int)

# check if there are any NaN values in gest_all
if test.isna().any().any():
    print("knn-set contains NaN elements")
else:
    print("knn-set does not contain NaN elements")

# Train a KNN model with default model hyperparameters
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(train, train_label)

# Make predictions on the testing set
pred = knn.predict(test)

print("we want the square of datapoints:", (len(test_label)) ** (1 / 2))

# use twice the square of the datapoints to look for the optimized k-value

######### OPTIMIZING ERROR MARIGN ######################################

############# k = squart(datapoints) #################
from sklearn.metrics import accuracy_score

# test the dp
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(train, train_label)
y_pred = knn.predict(test)
accuracy = accuracy_score(test_label, y_pred)
print('squared dp accuracy:', accuracy)

################## test various k ################################
from sklearn.metrics import mean_squared_error

# Try different values of k neighbors
ks = [x for x in range(1, 400) if x % 2 != 0]  # choose a range with uneven integers
errors = []

best_accuracy = 0
best_k = 0
# a loop that tests all k values and outputs the error
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train, train_label)
    y_pred = knn.predict(test)
    error = mean_squared_error(test_label, y_pred)
    errors.append(error)
    accuracy = accuracy_score(test_label, y_pred)
    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy
print('best accuracy:', best_accuracy)
print('best k:', best_k)

# Choose the optimal k (minimal error)
optimal_k = ks[np.argmin(errors)]
print(f'Optimal k: {optimal_k}')

'''
# Plot the error curve
import matplotlib.pyplot as plt
plt.plot(ks, errors)
plt.xlabel('k')
plt.ylabel('Error')
plt.title('Error Curve for knn Algorithm')
plt.show()
'''

#################### knn #######################################
# TA REDA PÅ k
# har för mig att det finns en regel att knn vill ha sqrt(datapoints) som k
# antingen om vi utgår från en algoritm som hittar minsta 'word'-arrayen och tar roten..
# ..ur den, det är illafall en början
# OM det inte fungerar, så kan vi ta sqrt(word_array)=k för varje ord, men blir...
# ... betydligt mindre optimerat, vilket knn redan är för stora mängder data
