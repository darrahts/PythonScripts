
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import tree, metrics, datasets, linear_model, preprocessing, cross_validation, svm
from pylab import rcParams
import os
import csv
import graphviz
import pydotplus
import collections

def LoadData(path, returnBunch=True):
    '''   This loads the data with map scores as targets  '''
    with open(os.path.join(path, 'BettysBrainDataForAnalysis.csv')) as f:
        file = csv.reader(f)

        #   get first line
        temp = next(file)
        numSamples = int(temp[0])
        numFeatures = int(temp[1])
        
        #   generate empty arrays
        data = np.empty((numSamples, numFeatures))
        targets = np.empty((numSamples,))

        #   get next line which is the column headers, less the last column (-1) i.e. feature names
        temp = next(file)
        featureNames = np.array(temp[:-1])

        for i, d in enumerate(file):
            data[i] = np.asarray(d[:-1], dtype=np.float64) #   all but last column are the data features
            targets[i] = np.asarray(d[-1], dtype=np.float64) #   only the last column are the targets

    if not returnBunch:
        return data, targets

    return datasets.base.Bunch(data=data, target=targets,feature_names=featureNames, DESCR="test description of the bettys brain  data")

def LoadData1(path, returnBunch=True):
    ''' This loads the data with group/individual as target '''
    with open(os.path.join(path, 'BettysBrainDataForAnalysis.csv')) as f:
        file = csv.reader(f)

        #   get first line
        temp = next(file)
        numSamples = int(temp[0])
        numFeatures = int(temp[1])
        
        #   generate empty arrays
        data = np.empty((numSamples, numFeatures))
        targets = np.empty((numSamples,))

        #   get next line which is the column headers, less the last column (-1) i.e. feature names
        temp = next(file)
        featureNames = np.array(temp[1:])

        for i, d in enumerate(file):
            data[i] = np.asarray(d[1:], dtype=np.float64) #   all but last column are the data features
            targets[i] = np.asarray(d[0], dtype=np.float64) #   only the last column are the targets

    if not returnBunch:
        return data, targets

    return datasets.base.Bunch(data=data, target=targets,feature_names=featureNames, DESCR="test description of the bettys brain  data")


def LoadData2(path):
    ''' This loads the data set as a nxm matrix '''
    with open(os.path.join(path, 'BettysBrainDataForAnalysis.csv')) as f:
        file = csv.reader(f)
        #   get first line
        temp = next(file)
        numSamples = int(temp[0])
        numFeatures = int(temp[1])
        
        #   generate empty arrays
        data = np.empty((numSamples, numFeatures))

        for i, d in enumerate(file):
            data[i] = np.asarray(d[:], dtype=np.float64) #
    return data

'''*****************************************************************'''
'''                                    REGRESSION                   '''
'''*****************************************************************'''
#   TO-DO polynomial regression, ridge regression, others

def Test_1(df, target): #   91.6% prediction accuracy i.e. percentage of variance the model explains uses least squares
    '''These values are highly related to final map score using statsmodels.api.OLS(target, df).fit() '''

    X = df[["num_map_moves", "time_viewing_concept_map_ms", "num_causal_link_adds", "num_causal_link_adds_effective"]]
    y = target["Final_Map_Score"]
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

def Test_2(df, target): #   69.5% prediction accuracy
    '''These values are not so much related to final map score'''

    X = df[["total_system_time_ms", "time_viewing_graded_questions_ms", "time_viewing_notes_ms", "time_viewing_science_book_ms"]]
    y = target["Final_Map_Score"]
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

def Test_3(df, target):#   92.9%
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
    Z = np.array(X["num_causal_link_adds_effective"])
    XX = []
    for val in Z:
        XX.append(int(val))
    Q = np.array(XX)
    y = np.array(y)
    yy = []
    for val in y:
        yy.append(int(val))
    s = np.array(yy)
    slope, intercept, rVal, pVal, stdErr = stats.linregress(Q, s)
    line = slope*Z+intercept
    plt.plot(Z, y, 'o', Z, line)
    plt.text(5, 22, "r value: {:.3f}".format(rVal))
    plt.text(5, 21, "std err: {:.3f}".format(stdErr))
    plt.text(5, 25, "Line of Best Fit")
    plt.text(5, 24, "y={:.2f}x + {:.2f}".format(slope, intercept) )#,fontdict=font)
    plt.title("Sample Plot of Betty's Brain Data Analysis ")
    plt.xlabel("Effective Link Adds")
    plt.ylabel("Final Map Score")
    #plt.scatter(Z, y)
    plt.show()

def Test_5(df, target): #   84.8%
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
    #   load the data
    data = LoadData("E:\Downloads")
    #   independent variables
    df = pd.DataFrame(data.data, columns=data.feature_names)
    #   dependent variable
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


'''*****************************************************************'''
'''                                    DECISION TREE                '''
'''*****************************************************************'''

def DecisionTreeTesting():
    #   load the data
    data = LoadData1("/home/tdarrah/PythonScripts/")
    df = pd.DataFrame(data.data, columns=data.feature_names)

    #   features
    X = df[["num_map_moves", "time_viewing_concept_map_ms", "num_causal_link_adds", "num_causal_link_adds_effective"]] 
    #X = df[["num_map_moves", "time_viewing_concept_map_ms", "time_viewing_graded_questions_ms", "time_viewing_graded_explanations_ms", "time_viewing_ungraded_explanations_ms", "num_causal_link_adds", "num_causal_link_adds_effective"]]
    #X = df

    #   target
    y = pd.DataFrame(data.target, columns=["subject_id"])
    
    #   split the data
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(X, y, test_size = .25, random_state=100)
    
    #   decision tree classifiers, _1 is gini, _2 is entropy, other parameters can be experimented with 
    dtc_1 = tree.DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=8, min_samples_split=6, min_samples_leaf=3)
    dtc_1.fit(xTrain, yTrain)
    yPred_1 = dtc_1.predict(xTest)
    print(metrics.accuracy_score(yTest,yPred_1))

    dtc_2 = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=8, min_samples_split=6, min_samples_leaf=3)
    dtc_2.fit(xTrain, yTrain)
    yPred_2 = dtc_2.predict(xTest)
    print(metrics.accuracy_score(yTest,yPred_2))

    #   decision tree visualization, change feature names to match the correct feature set above
    dot_data = tree.export_graphviz(dtc_2, feature_names=["num_map_moves", "time_viewing_concept_map_ms", "num_causal_link_adds", "num_causal_link_adds_effective"], out_file=None, filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
   
    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
 
    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png("dtc_2.png")
    
    #visualization - gives errors but should work
    #dot_data = tree.export_graphviz(dtc_2, out_file=None)
    #graph = graphviz.Source(dot_data)
    #graph.render("test2")


'''*****************************************************************'''
'''                                        MAIN                     '''
'''*****************************************************************'''

if __name__ == "__main__":
    #clear the warnings generated from the imported libraries
    os.system('cls' if os.name == 'nt' else 'clear')
    DecisionTreeTesting()
    #RegressionTesting()




