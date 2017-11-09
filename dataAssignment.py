
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets, linear_model, preprocessing, cross_validation, svm
import os
import csv

def LoadData(path, returnBunch=True):
    with open(os.path.join(path, 'BettysBrainDataForAnalysis.csv')) as f:
        file = csv.reader(f)

        #get first line
        temp = next(file)
        numSamples = int(temp[0])
        numFeatures = int(temp[1])
        
        #generate empty arrays
        data = np.empty((numSamples, numFeatures))
        targets = np.empty((numSamples,))

        #get next line which is the column headers, less the last column (-1) i.e. feature names
        temp = next(file)
        featureNames = np.array(temp[:-1])

        for i, d in enumerate(file):
            data[i] = np.asarray(d[:-1], dtype=np.float64) #all but last column are the data features
            targets[i] = np.asarray(d[-1], dtype=np.float64) #only the last column are the targets

    if not returnBunch:
        return data, targets

    return datasets.base.Bunch(data=data, target=targets,feature_names=featureNames, DESCR="test description of the bettys brain  data")


def Test_1(df, target): # 91.6% prediction accuracy i.e. percentage of variance the model explains uses least squares
    '''These values are highly related to final map score using statsmodels.api.OLS(target, df).fit() '''

    X = df[["num_map_moves", "time_viewing_concept_map_ms", "num_causal_link_adds", "num_causal_link_adds_effective"]]
    y = target["Final_Map_Score"]
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

def Test_2(df, target): # 69.5% prediction accuracy
    '''These values are not so much related to final map score'''

    X = df[["total_system_time_ms", "time_viewing_graded_questions_ms", "time_viewing_notes_ms", "time_viewing_science_book_ms"]]
    y = target["Final_Map_Score"]
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

def Test_3(df, target):# 92.9%
    '''These values are another combination'''

    X = df[["num_map_moves", "time_viewing_concept_map_ms", "time_viewing_graded_questions_ms", "time_viewing_graded_explanations_ms", "time_viewing_ungraded_explanations_ms", "num_causal_link_adds", "num_causal_link_adds_effective"]]
    y = target["Final_Map_Score"]
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())


def Test_4(df, target): # 94.7% 
    '''This is the entire data set'''
    X = df
    y = target
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

def Test_5(df, target): # 84.8%
    '''This uses the entire data set with sklearn.linear_model.LinearRegression().fit(df,target)'''
    model = linear_model.LinearRegression().fit(df, target)
    predictions = model.predict(df)
    
    print("ACTUAL\t\tPREDICTED")
    for i in range(len(predictions)):
        print(str(target["Final_Map_Score"][i]) + "\t\t" + str(predictions[i]))
    print("Score: " + str(model.score(df, target)))


def Test_6(df, target):
    '''uses svr and the entire dataset. I dont really understand the confidence values given.'''
    X = preprocessing.scale(np.array(df))
    y = np.array(target["Final_Map_Score"])
    trainX, testX, trainY, testY = cross_validation.train_test_split(X, y, test_size=.25)

    for type in ["linear", "poly", "rbf", "sigmoid"]:
        classifier = svm.SVR(kernel=type)
        print(classifier.fit(trainX, trainY).predict(testX))
        confidence = classifier.score(testX, testY)
        print(type + "\t" + str(confidence))


if __name__ == "__main__":
    #clear the warnings generated from the imported libraries
    os.system('cls' if os.name == 'nt' else 'clear')

    #load the data
    data = LoadData("C:\ProgramData\Anaconda3\Lib\site-packages\sklearn\datasets\data")
    #print(data.feature_names)
    #print(data.DESCR)

    #independent variables
    df = pd.DataFrame(data.data, columns=data.feature_names)
    #print(df["num_map_moves"])

    #dependent variable
    target = pd.DataFrame(data.target, columns=["Final_Map_Score"])
    #print(target["Final_Map_Score"][39])
    
    #z = df[df.columns[2:4]]
    #print(z)

    #Test_1(df, target)
    #Test_2(df, target)
    #Test_3(df, target)
    Test_4(df, target)
    #Test_5(df, target)
    #Test_6(df, target)
    
