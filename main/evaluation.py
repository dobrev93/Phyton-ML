from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#A file that contains all the functions that evalaute the predictions of a particular classifier

#A function that creates a confusion matrix given the test labels and the predicted lables
#@iput test_labels - An array of the actual labels of the data set
#@input predicted_labels - An array of the predicted labels for the data set
def confusion_matrix_results(test_labels, predicted_labels):
    return confusion_matrix(test_labels, predicted_labels)

#A function that creates a classification report given the test labels and the predicted lables
#@iput test_labels - An array of the actual labels of the data set
#@input predicted_labels - An array of the predicted labels for the data set
def classification_report_results(test_labels, predicted_labels):
    return classification_report(test_labels, predicted_labels)

def classification_accuracy_score(test_labels, predicted_labels):
    return accuracy_score(test_labels, predicted_labels)

##TO BE IMPLEMENTED
# Other manners of result evaluation
# A way to generate/store graphs that we can create from different evaluations

def storeEvaluationResults(evaluationData):
    return -1