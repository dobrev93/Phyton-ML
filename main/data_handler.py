#import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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