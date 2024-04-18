### EN TUTORIAL PÅ HUR MAN TRÄNAR KNNN ###
# tänker att vi gör om denna för vår egen data för att säkerställa att vår data kan hanteras av classifyers

# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Instantiate the KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = knn.predict(X_test)

# Model accuracy
accuracy = knn.score(X_test, y_test)
print('KNN model accuracy: ', accuracy)
type(iris)