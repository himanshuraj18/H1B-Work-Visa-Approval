#importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


"""
    Function to seperate X and Y from a dataset.
"""
def sep_X_Y(df):
    cols=df.columns
    X = df[cols[:-1]].to_numpy()
    Y = df[cols[-1]].to_numpy()
    return X,Y

"""
    Function to generate graph based on inputs given.
    Using this to plot Train and Validation loss for Hyper Parameter vs Accuracy.
    All the generated images will be store in img folder.
"""
def graph_plot(X,Y1,Y2,xlabel,info='',fname='img/img.png'):
    plt.plot(X,Y1,label='Train Accuracy')
    plt.plot(X,Y2,label='Validation Accuracy')
    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.title(info)
    plt.savefig(fname)


"""
    Applying Gaussian Naive Bayes Classifier.
    Output -
        1. Give Test Accuracy using the best model.
        2. Generate Confusion Matrix.
"""
def apply_Naive_Bayes(X_train, Y_train, X_test, Y_test):
    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    print("Train Accuracy =",clf.score(X_train,Y_train))
    print("Test Accuracy =",clf.score(X_test,Y_test))
    Y_pred = clf.predict(X_test)
    print(confusion_matrix(Y_test, Y_pred))


"""
    Applying KNN as Classifier with cv = 4.
    Doing Hyperparameter Tuning using Grid Search Technique to tune n_neighbors.
    Output -
        1. Will generate a table of all train and validation score corresponding to each set of hyperparameters.
        2. Give Test Accuracy using the best model.
        3. Generate Confusion Matrix.
        4. Generate Relevant Graphs. 
"""
def apply_KNN(X_train, Y_train, X_test, Y_test):
    params = {
        'n_neighbors' : [i for i in range(3,100,2)],
        'n_jobs' : [-1]
    }
    clf = GridSearchCV(KNeighborsClassifier(), param_grid = params, return_train_score = True, verbose = 5, n_jobs=-1, cv=4)
    clf.fit(X_train, Y_train)
    results = pd.DataFrame(clf.cv_results_)
    results = results[['param_n_neighbors','mean_train_score', 'mean_test_score']]
    print(results.to_string())
    graph_plot(params['n_neighbors'],results['mean_train_score'].to_numpy(),results['mean_test_score'].to_numpy(),'N-Neighbors','Accuracy vs N-Neighbors','img/n_neighbors.png')
    print("Test Accuracy =",clf.score(X_test,Y_test))
    Y_pred = clf.predict(X_test)
    print(confusion_matrix(Y_test, Y_pred))


"""
    Applying Logistic Regression as Classifier with cv = 4.
    Doing Hyperparameter Tuning using Grid Search Technique to tune max_iter.
    Output -
        1. Will generate a table of all train and validation score corresponding to each set of hyperparameters.
        2. Give Test Accuracy using the best model.
        3. Generate Confusion Matrix.
        4. Generate Relevant Graphs. 
"""
def apply_Logistic_Regression(X_train, Y_train, X_test, Y_test):
    params = {'max_iter' : [i for i in range(1,200)]}
    clf = GridSearchCV(LogisticRegression(), param_grid = params, return_train_score = True, verbose = 5, n_jobs=-1, cv=4)
    clf.fit(X_train,Y_train)
    results = pd.DataFrame(clf.cv_results_)
    results = results[['param_max_iter','mean_train_score', 'mean_test_score']]
    print(results.to_string())
    graph_plot(params['max_iter'],results['mean_train_score'].to_numpy(),results['mean_test_score'].to_numpy(),'Max Iterations','Accuracy vs Max Iterations','img/max_iters.png')
    print("Test Accuracy =",clf.score(X_test,Y_test))
    Y_pred = clf.predict(X_test)
    print(confusion_matrix(Y_test, Y_pred))

    
"""
    Applying Random Forest Classifier with cv = 4.
    Doing Hyperparameter Tuning using Grid Search Technique.
    Output -
        1. Will generate a table of all train and validation score corresponding to each set of hyperparameters.
        2. Give Test Accuracy using the best model.
        3. Generate Confusion Matrix.
        4. Generate Relevant Graphs. 
"""
def apply_RF(X_train, Y_train, X_test, Y_test):
    # params = {'n_estimators' : [i for i in range(100,501,100)],
    #         'max_depth' : [i for i in range(15,40)],
    #         'bootstrap' : [True, False],
    #         'max_features' : [i for i in range(5,25)],
    #         'min_samples_split' : [i for i in range(2,11,2)],
    #         'min_samples_leaf' : [i for i in range(1,6,2)],
    #         'random_state' : [0],
    #         'n_jobs' : [-1]
    #        }

    #using different set of parameters for Random Forest, following dictionary shows the best parameters

    params = {'n_estimators' : [50],
              'max_depth' : [26],
              'bootstrap' : [False],
              'max_features' : [9],
              'random_state' : [0],
              'n_jobs' : [-1]
             }
    clf = GridSearchCV(RandomForestClassifier(), param_grid = params, return_train_score = True, verbose = 100, n_jobs=-1, cv=4)
    clf.fit(X_train,Y_train)
    results = pd.DataFrame(clf.cv_results_)
    results = results[['mean_train_score', 'mean_test_score']]
    print(results.to_string())
    graph_plot(params['max_features'],results['mean_train_score'].to_numpy(),results['mean_test_score'].to_numpy(),'Max Features','Accuracy vs Max Features','img/max_features.png')
    print("Test Accuracy =",clf.score(X_test,Y_test))
    Y_pred = clf.predict(X_test)
    print(confusion_matrix(Y_test, Y_pred))

"""
    Applying Neural Network with cv = 4.
    Doing Hyperparameter Tuning using Grid Search Technique.
    Output -
        1. Will generate a table of all train and validation score corresponding to each set of hyperparameters.
        2. Give Test Accuracy using the best model.
        3. Generate Confusion Matrix.
        4. Generate Relevant Graphs. 
"""

def apply_Neural_Net(X, Y):
    params = {'random_state' : [0], 'hidden_layer_sizes':[(264, 64, 32)], 'max_iter':[500], 'alpha':[0.001]}
    params['activation'] = ['tanh','relu','logistic','identity']
    clf = GridSearchCV(MLPClassifier(), param_grid = params, return_train_score = True, verbose = 5, n_jobs = -1, cv = 4)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify = Y)
    clf.fit(X_train,Y_train)
    results = pd.DataFrame(clf.cv_results_)
    results = results[['params','mean_train_score','mean_test_score']]
    print (results.to_string())
    x = ['tanh', 'relu','logistic','identity']
    y1 = list(results['mean_train_score'])
    y2 = list(results['mean_test_score'])
    graph_plot(x, y1, y2, 'Activation Function', 'Accuracy vs Activation Function', 'img/act_func.png')
    ypred = clf.predict(X_test)
    print ("Accuracy on Testing Data is ",accuracy_score(ypred, Y_test))
    print (confusion_matrix(Y_test, ypred))
    return x, y1, y2



df = pd.read_csv('mkc.csv')                         #importing data
df = df.sample(frac=1).reset_index(drop=True)       #shuffling

X,Y = sep_X_Y(df)

#Stratified Sampling with ratio of 60:20:20 for Train:Validation:Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify = Y)

#applying models
apply_Logistic_Regression(X_train, Y_train, X_test, Y_test)

apply_Naive_Bayes(X_train, Y_train, X_test, Y_test)

apply_RF(X_train, Y_train, X_test, Y_test)

apply_KNN(X_train, Y_train, X_test, Y_test)