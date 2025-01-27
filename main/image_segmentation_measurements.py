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


def cancer_test(samplingType, normalizeType, selectBest, kBest=10):
    print("----------------------------Preprocessing------------------------------------------")
    start_preprocessing_time = time.time()
    train_url_cancer = '../data/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.lrn.csv'
    X, Y = getFeatureLabelData(train_url_cancer, 1)
    if (samplingType == 'over'):
        X_sampled, Y_sampled = overSampling(X, Y)
        x_train, x_test, y_train, y_test = trainTestSplit(X_sampled, Y_sampled)
    elif (samplingType == 'under'):
        X_sampled, Y_sampled = underSampling(X, Y)
        x_train, x_test, y_train, y_test = trainTestSplit(X_sampled, Y_sampled)
    elif (samplingType == 'combined'):
        X_sampled, Y_sampled = combinedSampling(X, Y)
        x_train, x_test, y_train, y_test = trainTestSplit(X_sampled, Y_sampled)
    else:
        x_train, x_test, y_train, y_test = trainTestSplit(X.values, Y)
    # Pre processing to be added here, not after the preprocessing_time
    if (normalizeType == 'minmax'):
        x_train = minMaxScailing(x_train)
        x_test = minMaxScailing(x_test)
    elif (normalizeType == 'normal'):
        x_train, x_test = normalizeData(x_train, x_test)
    if selectBest:
        x_train, x_test = selectKBest(x_train, y_train, x_test, y_test, kBest)
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
    print('The auc_roc score is: ', classification_roc_auc_score(y_test, dt_label_prediction, proba, 0))
    print("The prediction time for Decision tree is:", prediction_dt_time - start_prediction_dt_time)
    print("----------------------------end decisionTree------------------------------------------")

    print("-----------------------NaiveBayes-----------------------------------------")
    start_prediction_bayes_time = time.time()
    nb_label_prediction, proba = getPredictionData("NaiveBayes", x_train, x_test, y_train, y_test)
    prediction_bayes_time = time.time()
    print(confusion_matrix_results(y_test, nb_label_prediction))
    print(classification_report_results(y_test, nb_label_prediction))
    print("The accuracy is: ", classification_accuracy_score(y_test, nb_label_prediction))
    print('The auc_roc score is: ', classification_roc_auc_score(y_test, nb_label_prediction, proba, 0))
    print("The prediction time for Naive Bayes is:", prediction_bayes_time - start_prediction_bayes_time)
    print("-----------------------End NaiveBayes-----------------------------------------")

    print("----------------------------kNeighbours------------------------------------------")
    start_prediction_knn_time = time.time()
    kn_label_prediction, proba = getPredictionData("kNeighbours", x_train, x_test, y_train, y_test, 5)
    prediction_knn_time = time.time()
    print(confusion_matrix_results(y_test, kn_label_prediction))
    print(classification_report_results(y_test, kn_label_prediction))
    print('The accuracy is: ', classification_accuracy_score(y_test, kn_label_prediction))
    print('The auc_roc score is: ', classification_roc_auc_score(y_test, kn_label_prediction, proba, 0))
    print("The prediction time for kNN is:", prediction_knn_time - start_prediction_knn_time)
    print("----------------------------End kNeighbours------------------------------------------")


def main():
    """
    Main function.
    :return:
    """

    cancer_test('over', 'minmax', True, 20)


if __name__ == '__main__':
    main()
    sys.exit(0)