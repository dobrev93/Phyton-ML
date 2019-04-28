import sys
import time
import pandas
from data_handler import combinedSampling, underSampling, overSampling, read_data, getFeatureLabelData, trainTestSplit, \
    minMaxScailing, normalizeData, plotFeatureHistogram, plotMissingValuesHistogram, addNumericalMissingValueMean, \
    addNominalMissingValueMode, writeToCsv, getRowIDs, featureEncoding, featureOneHotEncoding, selectKBest, \
    selectRandomForests, selectDecisionTree
from classifier import getPredictionData
from evaluation import confusion_matrix_results, classification_report_results, classification_accuracy_score, \
    classification_roc_auc_score


def amazon_measure(scaling=False, sampling = "None", featureSelection = False):
    """
    Main function.
    :return:
    """

    print("----------------------------Preprocessing------------------------------------------")
    start_preprocessing_time = time.time()
    train_url = '../data/Amazon_Review_Data/amazon_review_ID.shuf.lrn.csv'
    X, Y = getFeatureLabelData(train_url, -1)

    if sampling == "Under":
        X_sampled, Y_sampled = underSampling(X, Y)
    elif sampling == "Over":
        X_sampled, Y_sampled = overSampling(X, Y)
    elif sampling == "Combined":
        X_sampled, Y_sampled = combinedSampling(X, Y)
    else:
        X_sampled, Y_sampled = X.values, Y


    if scaling:
       X_values = minMaxScailing(X_sampled)
    else:
       X_values = X_sampled






    x_train, x_test, y_train, y_test = trainTestSplit(X_sampled, Y_sampled)

    if featureSelection:
        x_train, x_test = selectKBest(x_train, y_train, x_test,  y_test)



    # Pre processing to be added here, not after the preprocessing_time

    preprocessing_time = time.time()
    print("The processing time is: ", preprocessing_time - start_preprocessing_time)
    print("----------------------------end Preprocessing------------------------------------------")

    print("----------------------------decisionTree------------------------------------------")
    start_prediction_dt_time = time.time()
    dt_label_prediction, proba = getPredictionData("decisionTree", x_train, x_test, y_train, y_test)
    prediction_dt_time = time.time()
    print(confusion_matrix_results(y_test, dt_label_prediction))
    print(classification_report_results(y_test, dt_label_prediction))
    print("The accuracy is: ", classification_accuracy_score(y_test, dt_label_prediction))
    #print('The auc_roc score is: ', classification_roc_auc_score(y_test, dt_label_prediction, proba, 0))
    print("The prediction time for Decision tree is:", prediction_dt_time - start_prediction_dt_time)
    print("----------------------------end decisionTree------------------------------------------")

    print("-----------------------NaiveBayes-----------------------------------------")
    start_prediction_bayes_time = time.time()
    nb_label_prediction, proba = getPredictionData("NaiveBayes", x_train, x_test, y_train, y_test)
    prediction_bayes_time = time.time()
    print(confusion_matrix_results(y_test, nb_label_prediction))
    print(classification_report_results(y_test, nb_label_prediction))
    print("The accuracy is: ", classification_accuracy_score(y_test, nb_label_prediction))
    #print('The auc_roc score is: ', classification_roc_auc_score(y_test, nb_label_prediction, proba, 0))
    print("The prediction time for Naive Bayes is:", prediction_bayes_time - start_prediction_bayes_time)
    print("-----------------------End NaiveBayes-----------------------------------------")

    print("----------------------------kNeighbours------------------------------------------")
    start_prediction_knn_time = time.time()
    kn_label_prediction, proba = getPredictionData("kNeighbours", x_train, x_test, y_train, y_test, 5)
    prediction_knn_time = time.time()
    print(confusion_matrix_results(y_test, kn_label_prediction))
    print(classification_report_results(y_test, kn_label_prediction))
    print('The accuracy is: ', classification_accuracy_score(y_test, kn_label_prediction))
    #print('The auc_roc score is: ', classification_roc_auc_score(y_test, kn_label_prediction, proba, 0))
    print("The prediction time for kNN is:", prediction_knn_time - start_prediction_knn_time)
    print("----------------------------End kNeighbours------------------------------------------")

def main():

    amazon_measure(sampling="Combined", scaling= True)

if __name__ == '__main__':
        main()
        sys.exit(0)