#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA



from normalizer import normalizeStandardScale
#A file that deals with data loading and pre-processing


#Function that loads the data from a url
#@input url - The location of the file
#@return The dataset in the file
def read_data(url):
    dataset = pd.read_csv(url)
    return dataset

#Creates and saves a histogram for a particular feature of the data set.
#@input datasetFeature - The feature that you want to create a histogram. THE INPUT FORMAT IS: "dataset.FeatureName"
#@input pic_name - The name of the picture generated
def plotFeatureHistogram(datasetFeature, pic_name):
    plt.figure(figsize=(15,8))
    plt.title(pic_name)
    temp = sns.distplot(datasetFeature, bins =30)
    plt.savefig("./histograms/" + pic_name + ".png")

#Creates and saves a histogram for all the missing values of the data set.
#@input dataset - The dataset that you want to create a histogram.
#@input pic_name - The name of the picture generated
def plotMissingValuesHistogram(dataset, pic_name):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    f, ax = plt.subplots(figsize=(12, 5))
    plt.figure(figsize=(30,22))
    plt.xticks(rotation='90')
    sns.barplot(x=missing_data.index, y=missing_data['Percent'])
    missing_data.head()
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.savefig("./histograms/" + pic_name + ".png")

#Drops all the rows in a dataset that have at least one missing value
#@input dataset - The dataset.
#@return The updated dataset
def dropRow(dataset):
    dataset.dropna(inplace=True)
    return dataset

#Drops all the rows in a dataset that reach a threshold of missing values
#@input dataset - The dataset.
#@input threshold - The number of missing features that a row needs to reach in order to be deleted
#@return The updated dataset
def dropRowThreshold(dataset, threshold):
    dataset.dropna(thresh=threshold,inplace=True)
    return dataset

#A function that adds the mean of a feature to all the missing values that that feature has.
#@input datasetFeature - An array of all the values of a feature
def addNumericalMissingValueMean(datasetFeature):
    datasetFeature.fillna(datasetFeature.mean(),inplace=True)

def addNominalMissingValueMode(datasetFeature):
    #to be implemented
    return -1

def getFeatureLabelData(train_url,classColumn ):
    dataset = read_data(train_url)
    X = dataset.drop(dataset.columns[[0, classColumn]], axis=1)
    Y = dataset.iloc[:, classColumn].values
    return X,Y

def getRowIDs(data_url, IDcolumn):
    dataset = read_data(data_url)
    IDs = dataset.iloc[:, IDcolumn].values
    return IDs


def minMaxScailing(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    return X

def normalizeData(X_train, X_test):
    normalized_data = normalizeStandardScale(X_train, X_test)
    normalized_training_data = normalized_data[0]
    normalized_test_data = normalized_data[1]
    return normalized_training_data, normalized_test_data

def trainTestSplit(X, Y):
    cv = StratifiedKFold(n_splits=4)
    for train_index, test_index in cv.split(X, Y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

    return X_train, X_test, y_train, y_test

#Ideas for implementing the label encoder
#Go thourgh all the columns, and if a column does not have numerical values perform label encoding
#Look for label encoding for the dataset
def featureEncoding(train_dataset, test_dataset, columns):
    le = LabelEncoder()
    for column in columns: 
        train_dataset[:, column] = le.fit_transform(train_dataset[:, column])
    for column in columns: 
        test_dataset[:, column] = le.fit_transform(test_dataset[:, column])
    return train_dataset, test_dataset

def featureOneHotEncoding(traindataset, testdataset):
    ohe = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
    x_train = ohe.fit_transform(traindataset)
    x_test = ohe.fit_transform(testdataset)
    return x_train, x_test

def writeToCsv(filename, IDs, label_prediction):
    zippedVal = zip(IDs, label_prediction)
    with open(filename, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['ID', 'class'])
        for element in zippedVal:
            filewriter.writerow([element[0] , element[1]])

def overSampling(X, Y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    return X_resampled, y_resampled

def underSampling(X, Y):
    nm1 = NearMiss(version=1)
    X_resampled, y_resampled = nm1.fit_resample(X, Y)
    return X_resampled, y_resampled

def combinedSampling(X, Y):
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, Y)
    return X_resampled, y_resampled


def selectKBest(x_train_ds, y_train_ds, x_test_ds, y_test_ds, kBest):
    bestfeatures = SelectKBest(score_func=chi2, k=kBest)
    x_train = bestfeatures.fit_transform(x_train_ds,y_train_ds)
    x_test = bestfeatures.fit_transform(x_test_ds,y_test_ds)
    return x_train, x_test

def selectRandomForests(x_train_ds, y_train_ds, x_test_ds, y_test_ds, max_features):
    x_train = SelectFromModel(RandomForestClassifier(n_estimators = 100), max_features=max_features)
    x_train = x_train.fit_transform(x_train_ds, y_train_ds)
    x_test = SelectFromModel(RandomForestClassifier(n_estimators = 100), max_features=max_features)
    x_test = x_test.fit_transform(x_test_ds, y_test_ds)
    return x_train, x_test

def selectDecisionTree(x_train_ds, y_train_ds, x_test_ds, y_test_ds, max_features):
    x_train = SelectFromModel(ExtraTreesClassifier(n_estimators = 100), max_features=max_features)
    x_train = x_train.fit_transform(x_train_ds, y_train_ds)
    x_test = SelectFromModel(ExtraTreesClassifier(n_estimators = 100), max_features=max_features)
    x_test = x_test.fit_transform(x_test_ds, y_test_ds)
    return x_train, x_test

def PCA(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    return pca.explained_variance_ratio_
