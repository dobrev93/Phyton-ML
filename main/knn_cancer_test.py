import sys

from main.data_handler import read_data, getFeatureLabelData, trainTestSplit, minMaxScailing, normalizeData, plotFeatureHistogram, plotMissingValuesHistogram, addNumericalMissingValueMean, \
    addNominalMissingValueMode
from main.classifier import kNeighbours, naiveBayes, decisionTree
from main.evaluation import confusion_matrix_results, classification_report_results, classification_accuracy_score


# A basic example of how a test case should look like
train_url_cancer = '../data/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.lrn.csv'
test_url_cancer = '../data/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.tes.csv'
test_label_url_cancer = '../data/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.sol.ex.csv'

train_url_amazon = '../data/Amazon_Review_Data/amazon_review_ID.shuf.lrn.csv'
test_url_amazon = '../data/Amazon_Review_Data/amazon_review_ID.shuf.tes.csv'
test_label_url_amazon = '../data/Amazon_Review_Data/amazon_review_ID.shuf.sol.ex.csv'




def getPredictionData(type, X_train, X_test, Y_train, Y_test):

    if (type == "NaiveBayes"):
        label_prediction = naiveBayes(X_train,Y_train, X_test)
    elif (type == "kNeighbours"):
        label_prediction = kNeighbours(5, X_train, Y_train, X_test)
    else:
        label_prediction = decisionTree(X_train,Y_train, X_test)

    #print(label_prediction)
    #print(confusion_matrix_results(Y_test, label_prediction))
    #print(classification_report_results(Y_test, label_prediction))
    print(classification_accuracy_score(Y_test, label_prediction))


def main():
    """
    Main function.
    :return:
    """

    print("-----------------------NaiveBayes-----------------------------------------")
    X,Y = getFeatureLabelData(train_url_amazon, -1)
    #If we want to test scaled data uncomment X_values and replace x.values with x_values
    X_values = minMaxScailing(X)

    x_train, x_test, y_train, y_test = trainTestSplit(X_values, Y)

    #if we want to normalize data remove next comment
    #x_train, x_test = normalizeData(x_train, x_test)
    getPredictionData("NaiveBayes", x_train, x_test, y_train, y_test)
    print("----------------------------kNeighbours------------------------------------------")
    getPredictionData("kNeighbours", x_train, x_test, y_train, y_test)
    print("----------------------------decisionTree------------------------------------------")
    getPredictionData("decisionTree", x_train, x_test, y_train, y_test)


    # Loading the training and testing dataset, as well as removing not abundant values (might have to perform some missing values check, and maybe add/remove values based on that)
    # A good way to check for missing values, is to print dataset.info().
    # Moreover you can use the plotMissingValuesHistogram which will generate a histogram with the percentage of missing features
    # For the
    #dataset = read_data(train_url)
    #dataset.info()
    #X_train1 = dataset.iloc[:, 2:].values
    #y_train1 = dataset.iloc[:, 1].values
    #testset = read_data(test_url)
    #X_test1 = testset.iloc[:, 1:].values
    #testlabels = read_data(test_label_url)
    #y_test1 = testlabels.iloc[:, 1].values

    # Data normalization, still to implement different types of scaling/reducing value features.
    # Also need to create manners of "normalizing" nominal values
    #normalized_data = normalizeStandardScale(X_train1, X_test1)
    #normalized_training_data = normalized_data[0]
    #normalized_test_data = normalized_data[1]

    # HOCUS POCUS Predict!!!
    #label_prediction = kNeighbours(3, normalized_training_data, y_train1, normalized_test_data)

    #store the evaluation results
    #it needs to be stressed that the results needs to be stored not printed in the console/terminal but the implementation is not finished yet.
    #print(confusion_matrix_results(y_test1, label_prediction))
   # print(classification_report_results(y_test1, label_prediction))


if __name__ == '__main__':
    main()
    sys.exit(0)


