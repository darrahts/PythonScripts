
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import tree, metrics, datasets, linear_model, preprocessing, cross_validation, svm
from pylab import rcParams
import os
import csv
import graphviz

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

def LoadData1(path, returnBunch=True):
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
        featureNames = np.array(temp[1:])

        for i, d in enumerate(file):
            data[i] = np.asarray(d[1:], dtype=np.float64) #all but last column are the data features
            targets[i] = np.asarray(d[0], dtype=np.float64) #only the last column are the targets

    if not returnBunch:
        return data, targets

    return datasets.base.Bunch(data=data, target=targets,feature_names=featureNames, DESCR="test description of the bettys brain  data")

'''                                    REGRESSION                   '''
# TO-DO polynomial regression, ridge regression, others

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

def RegressionTesting():
    #load the data
    data = LoadData("E:\Downloads")
    #independent variables
    df = pd.DataFrame(data.data, columns=data.feature_names)
    #dependent variable
    target = pd.DataFrame(data.target, columns=["Final_Map_Score"])

    Test_1(df, target)
    #Test_2(df, target)
    #Test_3(df, target)
    #Test_4(df, target)
    #Test_5(df, target)
    #Test_6(df, target)

    #print(df["num_map_moves"])
    #print(df)
    #print(target["Final_Map_Score"][39])
    #print(data.feature_names)
    print(data.DESCR)


'''                                    DECISION TREE                   '''


def DecisionTreeTesting():
    #load the data
    data = LoadData1("E:\Downloads")
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    X = df[["num_map_moves", "time_viewing_concept_map_ms", "num_causal_link_adds", "num_causal_link_adds_effective"]] 
    #X = df[["num_map_moves", "time_viewing_concept_map_ms", "time_viewing_graded_questions_ms", "time_viewing_graded_explanations_ms", "time_viewing_ungraded_explanations_ms", "num_causal_link_adds", "num_causal_link_adds_effective"]]
    #X = df

    y = pd.DataFrame(data.target, columns=["subject_id"])
    #split the data
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(X, y, test_size = .25, random_state=100)
    #decision tree classifier
    dtc_1 = tree.DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_split=6, min_samples_leaf=3)
    dtc_1.fit(xTrain, yTrain)
    yPred_1 = dtc_1.predict(xTest)
    print(yPred_1)
    print(metrics.accuracy_score(yTest,yPred_1))

    dtc_2 = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=5, min_samples_split=6, min_samples_leaf=3)
    dtc_2.fit(xTrain, yTrain)
    yPred_2 = dtc_2.predict(xTest)
    print(metrics.accuracy_score(yTest,yPred_2))

    #visualization - gives errors but should work
    dot_data = tree.export_graphviz(dtc_1, out_file=None)
    #graph = graphviz.Source(dot_data)
    #graph.render("test")




if __name__ == "__main__":
    #clear the warnings generated from the imported libraries
    os.system('cls' if os.name == 'nt' else 'clear')
    DecisionTreeTesting()


    #print(X)
    #print(Y)


