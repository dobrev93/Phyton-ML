from sklearn.preprocessing import StandardScaler 

#A file that contains functions for normalizing the values of features of a particular data set.

# A function that changes the values to a standard scaler
#@input train_set - An array of all the train_set values
#@input test_set - An array of all the test set values
def normalizeStandardScale(train_set, test_set):
    scaler = StandardScaler()  
    scaler.fit(train_set)
    X_train = scaler.transform(train_set)  
    X_test = scaler.transform(test_set)
    return [X_train, X_test]


##TO BE IMPLEMENTED
# Other ways of normalizing the data