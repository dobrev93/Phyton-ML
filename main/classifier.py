from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#A file that contains functions for each classifier that we will implement

#The function kNeighbours implements the KNN classifier
#@input n_neighbors - The number of neighbours the algorithm considers
#@input feature_train - The processed data set
#@input label_train - The labels of the data set
#@input feature_test - The processed test set
#@return The prediction


def kNeighbours(n_neighbors, feature_train, label_train, feature_test):
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(feature_train, label_train)
    label_prediction = classifier.predict(feature_test)
    return label_prediction

def naiveBayes(dataset):
    model = GaussianNB()
    model.fit(dataset.data, dataset.target)
    predicted = model.predict(dataset.data)
    return predicted
