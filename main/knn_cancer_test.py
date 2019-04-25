import sys

from main.normalizer import normalizeStandardScale
from main.data_handler import read_data, plotFeatureHistogram, plotMissingValuesHistogram, addNumericalMissingValueMean, \
    addNominalMissingValueMode
from main.classifier import kNeighbours
from main.evaluation import confusion_matrix_results, classification_report_results


def main():
    """
    Main function.
    :return:
    """
    # A basic example of how a test case should look like
    train_url = '../data/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.lrn.csv'
    test_url = '../data/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.tes.csv'
    test_label_url = '../data/184702-tu-ml-ss-19-breast-cancer/breast-cancer-diagnostic.shuf.sol.ex.csv'

    # Loading the training and testing dataset, as well as removing not abundant values (might have to perform some missing values check, and maybe add/remove values based on that)
    # A good way to check for missing values, is to print dataset.info().
    # Moreover you can use the plotMissingValuesHistogram which will generate a histogram with the percentage of missing features
    # For the
    dataset = read_data(train_url)
    dataset.info()
    X_train1 = dataset.iloc[:, 2:].values
    y_train1 = dataset.iloc[:, 1].values
    testset = read_data(test_url)
    X_test1 = testset.iloc[:, 1:].values
    testlabels = read_data(test_label_url)
    y_test1 = testlabels.iloc[:, 1].values

    # Data normalization, still to implement different types of scaling/reducing value features.
    # Also need to create manners of "normalizing" nominal values
    normalized_data = normalizeStandardScale(X_train1, X_test1)
    normalized_training_data = normalized_data[0]
    normalized_test_data = normalized_data[1]

    # HOCUS POCUS Predict!!!
    label_prediction = kNeighbours(3, normalized_training_data, y_train1, normalized_test_data)

    # store the evaluation results
    # it needs to be stressed that the results needs to be stored not printed in the console/terminal but the implementation is not finished yet.
    print(confusion_matrix_results(y_test1, label_prediction))
    print(classification_report_results(y_test1, label_prediction))


if __name__ == '__main__':
    main()
    sys.exit(0)


