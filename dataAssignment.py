
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import tree, metrics, datasets, linear_model, preprocessing, cross_validation, svm
from pylab import rcParams
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import os
import csv
import graphviz
import pydotplus
import collections
#from tabulate import tabulate
from prettytable import PrettyTable

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
    with open(os.path.join(path, 'BettysBrainDataForAnalysisAll3.csv')) as f:
        file = csv.reader(f)

        #   get first line
        #temp = next(file)
        #numSamples = int(temp[0])
        #numFeatures = int(temp[1])
        
        #   generate empty arrays
        #data = np.empty((numSamples, numFeatures))
        #targets = np.empty((numSamples,))

        data = np.empty((40, 15))
        targets = np.empty((40,))
        
        #   get next line which is the column headers, less the last column (-1) i.e. feature names
        temp = next(file)
        featureNames = np.array(temp[1:])

        for i, d in enumerate(file):
            data[i] = np.asarray(d[1:], dtype=np.float64) #   all but first column are the data features
            targets[i] = np.asarray(d[0], dtype=np.float64) #   only the first column are the targets

    if not returnBunch:
        return data, targets

    return datasets.base.Bunch(data=data, target=targets,feature_names=featureNames, DESCR="test description of the bettys brain  data")


def LoadData2(path):
    ''' This loads the data set as a nxm matrix '''
    with open(os.path.join(path, 'BettysBrainDataForAnalysisAll3.csv')) as f:
        file = csv.reader(f)
        #   get first line
        temp = next(file)
        features = np.array(temp[:])
        
        #   generate empty arrays, change this to match number of freatures
        data = np.empty((40, 16))

        for i, d in enumerate(file):
            data[i] = np.asarray(d[:], dtype=np.float64) #
    return data, features

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

    #   Plot, not sure why but the xx and yy q z etc made it work, should have worked with one conversion
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
    #X = df[["num_map_moves", "time_viewing_concept_map_ms", "num_causal_link_adds", "num_causal_link_adds_effective"]] 
    #X = df[["num_map_moves", "time_viewing_concept_map_ms", "time_viewing_graded_questions_ms", "time_viewing_graded_explanations_ms", "time_viewing_ungraded_explanations_ms", "num_causal_link_adds", "num_causal_link_adds_effective"]]
    X = df

    #   target
    y = pd.DataFrame(data.target, columns=["subject_id"])
    
    #   split the data
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(X, y, test_size = .27, random_state=100)

    
    #   decision tree classifiers, _1 is gini, _2 is entropy, other parameters can be experimented with
    dtcGini = tree.DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=50, min_samples_split=4, min_samples_leaf=1)
    dtcGini.fit(xTrain, yTrain)
    yPred_1 = dtcGini.predict(xTest)
    accuracyGini = metrics.accuracy_score(yTest,yPred_1)
    print(accuracyGini)

    #   decision tree visualization, change feature names to match the correct feature set above 
    dot_data_gini = tree.export_graphviz(dtcGini, feature_names=data.feature_names, class_names=["individual", "group"], out_file=None, filled=True, rounded=True)
    #dot_data_gini = tree.export_graphviz(dtcGini, feature_names=["num_map_moves", "time_viewing_concept_map_ms", "num_causal_link_adds", "num_causal_link_adds_effective"], class_names=["individual", "group"], out_file=None, filled=True, rounded=True)
    graph_gini = pydotplus.graph_from_dot_data(dot_data_gini)
   
    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph_gini.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
 
    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph_gini.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph_gini.write_png("dtcGini_FullDataSet_{:.2f}.png".format(accuracyGini))


    dtcEntropy = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=50, min_samples_split=4, min_samples_leaf=1)
    dtcEntropy.fit(xTrain, yTrain)
    yPred_2 = dtcEntropy.predict(xTest)
    accuracyEntropy = metrics.accuracy_score(yTest,yPred_2)
    print(accuracyEntropy)

    #   decision tree visualization, change feature names to match the correct feature set above 
    dot_data_Entropy = tree.export_graphviz(dtcEntropy, feature_names=data.feature_names, class_names=["individual", "group"], out_file=None, filled=True, rounded=True)
    #dot_data_Entropy = tree.export_graphviz(dtcEntropy, feature_names=["num_map_moves", "time_viewing_concept_map_ms", "num_causal_link_adds", "num_causal_link_adds_effective"], class_names=["individual", "group"], out_file=None, filled=True, rounded=True)
    graph_Entropy = pydotplus.graph_from_dot_data(dot_data_Entropy)
   
    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph_Entropy.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
 
    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph_Entropy.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph_Entropy.write_png("dtcEntropy_FullDataSet_{:.2f}.png".format(accuracyEntropy))

    


def RangeNormalization(X):
    temp = X
    i = 0
    minVal = min(X)
    maxVal = max(X)
    for val in X:
        temp[i] = ((val - minVal) / (maxVal - minVal))
        i += 1
    return temp


'''*****************************************************************'''
'''                                     CLUSTERING                  '''
'''*****************************************************************'''
def Clustering():
    data, features = LoadData2("/home/tdarrah/Documents/PythonScripts/")

    #normalize the millisecond values only
    for i in range(1, 16):
        X = data[:,i]
        data[:,i] = RangeNormalization(X)

    #   find the best method/metric combo for heirarchical clustering
    methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    metrics = ["euclidean", "cityblock", "cosine"]
    best = ("", "", 0.1)

    table = PrettyTable(["method", "metric", "cophenet"])
    for m in methods:
        for t in metrics:
            try:
                linkMatrix = linkage(data, method=m, metric=t)
                c, cophDists = cophenet(linkMatrix, pdist(data))
                if(c > best[2]):
                    best = (m, t, c)

                table.add_row([m, t, c])
                
                #   plot the dendrograms
                plt.figure(figsize=(25, 10))
                plt.title("Hierarchical Clustering Dendrogram using {} and {}".format(m, t))
                plt.xlabel("data index (0-39)")
                plt.ylabel("distance")
                dendrogram(linkMatrix, leaf_rotation = 90., leaf_font_size=8.)
                plt.show()
                
            except ValueError: 
                pass

    print("clustering data normalized to 0-1")
    print(table)
    print("best: {}".format(best))
    


'''*****************************************************************'''
'''                                        MAIN                     '''
'''*****************************************************************'''

if __name__ == "__main__":
    #clear the warnings generated from the imported libraries
    os.system('cls' if os.name == 'nt' else 'clear')

    DecisionTreeTesting()
    #RegressionTesting()
    #Clustering()
    



















