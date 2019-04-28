import sys

from data_handler import combinedSampling, underSampling, overSampling, read_data, getFeatureLabelData, trainTestSplit, minMaxScailing, normalizeData, plotFeatureHistogram, plotMissingValuesHistogram, addNumericalMissingValueMean, \
    addNominalMissingValueMode, writeToCsv, getRowIDs, featureEncoding, featureOneHotEncoding, selectKBest, selectRandomForests, selectDecisionTree
from classifier import getPredictionData
from evaluation import confusion_matrix_results, classification_report_results, classification_accuracy_score, roc_auc_score_results
import time
import pandas
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

# A basic example of how a test case should look like
train_url_cancer = '../data/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.lrn.csv'
test_url_cancer = '../data/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.tes.csv'
test_label_url_cancer = '../data/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.sol.ex.csv'

train_url_amazon = '../data/Amazon_Review_Data/amazon_review_ID.shuf.lrn.csv'
test_url_amazon = '../data/Amazon_Review_Data/amazon_review_ID.shuf.tes.csv'
test_label_url_amazon = '../data/Amazon_Review_Data/amazon_review_ID.shuf.sol.ex.csv'


train_url_image_segmentation = "../data/Image_Segmentation_Data/segmentation.data"





def main():
    """
    Main function.
    :return:
    """

    dataBank = pandas.read_csv('../data/Bank_Marketing_Data/bank-full.csv', sep=";")
    X = dataBank.drop(dataBank.columns[[10, -1]], axis=1)
    Y = dataBank.iloc[:, -1].values

    #X,Y = getFeatureLabelData(train_url_cancer, 1)
    #If we want to test scaled data uncomment X_values and replace x.values with x_values
    #X_values = minMaxScailing(X)

    #X_sampled, Y_sampled = underSampling(X, Y)
    #x_train, x_test, y_train, y_test = trainTestSplit(X_sampled, Y_sampled)

    x_train, x_test, y_train, y_test = trainTestSplit(X.values, Y)

    #if we want to normalize data remove next comment
    #x_train = minMaxScailing(x_train)
    #x_test = minMaxScailing(x_test)
    print(x_train)
    print("-----------------------Feature Encoding-----------------------------------------")
    x_train, x_test = featureEncoding(x_train, x_test, [1, 2, 3, 4, 6, 7, 8, 9, 10, 14])
    print("-----------------------After Encoding-----------------------------------------")
    #x_train_oec, x_test_oec = featureOneHotEncoding(x_train[:,[1, 2, 3, 4, 6, 7, 8, 9, 10, 14]], x_test)

    #x_train = x_train.drop(x_train.columns[1, 2, 3, 4, 6, 7, 8, 9, 10, 14])
    print("-----------------------After OneHot Encoding-----------------------------------------")
    print(x_train)
    #if we want to normalize data remove next comment
    #x_train = minMaxScailing(x_train)
    #x_test = minMaxScailing(x_test)

    #print("-----------------------Feature Selection-----------------------------------------")
    #print(x_test)
    #x_train, x_test = selectDecisionTree(x_train, y_train, x_test, y_test)
    #print("-----------------------After Selection-----------------------------------------")
    #print(x_test)

    #print("-----------------------NaiveBayes-----------------------------------------")
    #nb_label_prediction = getPredictionData("NaiveBayes", x_train, x_test, y_train, y_test)
    #print(confusion_matrix_results(y_test, nb_label_prediction))
    #print(classification_report_results(y_test, nb_label_prediction))
    #print(classification_accuracy_score(y_test, nb_label_prediction))
    #print("----------------------------kNeighbours------------------------------------------")
    #kn_label_prediction = getPredictionData("kNeighbours", x_train, x_test, y_train, y_test, 5)
    #print(confusion_matrix_results(y_test, kn_label_prediction))
    #print(classification_report_results(y_test, kn_label_prediction))
    #print(classification_accuracy_score(y_test, kn_label_prediction))
    #print("----------------------------decisionTree------------------------------------------")
    #dt_label_prediction = getPredictionData("decisionTree", x_train, x_test, y_train, y_test)
    #temp = confusion_matrix_results(y_test, dt_label_prediction)
    #print(classification_report_results(y_test, dt_label_prediction))
    #print(classification_accuracy_score(y_test, dt_label_prediction))

    #end = time.time()

    #print("The runtime is: " ,end-start)


    #print(dt_label_prediction)
    #print(len(dt_label_prediction))
    #IDs = getRowIDs(train_url_cancer, 0, len(dt_label_prediction))
    #print(IDs)
    #writeToCsv("test.csv", IDs, dt_label_prediction)


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


