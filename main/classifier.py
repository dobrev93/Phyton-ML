from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
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

def naiveBayes(feature_train, label_train, feature_test):
    classifier = GaussianNB()
    classifier.fit(feature_train, label_train)
    predicted = classifier.predict(feature_test)
    return predicted

def decisionTree(feature_train, label_train, feature_test):
    classifier = DecisionTreeClassifier()
    classifier.fit(feature_train, label_train)
    predicted = classifier.predict(feature_test)
    return predicted


def getPredictionData(type, X_train, X_test, Y_train, Y_test, N_NEIGHBORS=3):
    if (type == "NaiveBayes"):
        label_prediction = naiveBayes(X_train,Y_train, X_test)
    elif (type == "kNeighbours"):
        label_prediction = kNeighbours(N_NEIGHBORS, X_train, Y_train, X_test)
    else:
        label_prediction = decisionTree(X_train,Y_train, X_test)
    
    return label_prediction