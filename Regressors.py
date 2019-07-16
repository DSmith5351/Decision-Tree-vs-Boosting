# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:15:53 2019

@author: D2
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor 
GitDir = "C:/Users/D2/Documents/Github/Regressors/"

rng = np.random.RandomState(2)

def target_function(x):
    fast_oscillation = 0.5*np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    return slow_oscillation + fast_oscillation

def generate_one_data_sample(len_interval, num_features, sigma=0.1):
    '''
    Args:
        len_interval - the x values are uniformly distributed in the interval
        [0, len_interval]
        num_features - the number of (x,y) values
        sigma - the standard deviation of the y-values
        
    Returns: a tuple consisting of two arrays:
        x is the np.array of num_features numbers
        y is the np.array of corresponding y values
        
        One data sample is one vector of x values and one vector of y values
    '''
    x = len_interval * rng.rand(num_features)
    x = np.sort(x)
    noise = sigma * rng.randn(len(x)) #randn is N(0,1)
    y = target_function(x) + noise
    return(x,y)
    
def generate_data_samples(num_samples,
                          len_interval, 
                          num_features, 
                          sigma):
    '''
    Args:
        num_samples - the number of x-arrays and y-arrays produced by one 
        iteration of generate_one_data_sample
        len_interval - the x values are uniformly distributed in the interval
        [0, len_interval]
        num_features - the number of (x,y) values
        sigma - the standard deviation of the y-values
        
    Return: two matrices, one whose rows are samples of x and the other whose
    rows are samples of y
    '''
    data = [generate_one_data_sample(len_interval, num_features,sigma) 
            for i in range(num_samples) ]
    x_list, y_list = list(zip(*data))
    return (np.array(x_list), np.array(y_list))
    

def decision_tree_regressor_one_sample(x,y,depth):
    '''
    Args:
        x, y - one data sample
        depth - the maximum depth the decision tree should go.
    '''
    dt = DecisionTreeRegressor(random_state=0, max_depth=depth)
    dt.fit(x[:, None], y)
    return dt
    
def GradientBoostingRegressor_one_sample(x,y,depth):   
     '''
     See decision_tree_regressors
     '''
     gb = GradientBoostingRegressor(n_estimators=2**depth, 
                                   learning_rate=1.0, 
                                   max_depth=1)
     
     gb.fit(x[:, None], y)
     return gb
 
def regressor_list(reg,X,Y,depth):
    '''
    Args:
        X, Y - two matrices, one row from each constitutes on data sample
        xfit - as in decision_tree_regressor_one_sample
        depth - as in decision_tree_regressor_one_sample
        
    Return: a list of regressors fitted to x. Each row 
    comes from one data sample.
    '''
    return [reg(x,y,depth) for x,y in zip(X,Y)]


    
def make_predictions(regressors,xfit):
    '''
    Inputs - a list of regressors and xfit
    
    Return: np.array converts the list of fitted regressors into a matrix
    with one row per sample
    '''
    return np.array([r.predict(xfit[:, None]) for r in regressors])

def graph_data(x,y,xfit):
    f = plt.figure(figsize=[16,8])
    plt.scatter(x,y, c='g',s=16)
    plt.plot(xfit, target_function(xfit), '-k', alpha=0.7,
            linestyle = 'dashed');
    plt.title('Target Function and a Data Set')
    f.savefig(GitDir + 'Target Function and a Data Set.pdf',bbox_inches = "tight")
    
def graph(x,y,xfit,regressor,model_name):
    pred = regressor.predict(xfit[:, None])
    f = plt.figure(figsize=[16,8])
    plt.scatter(x,y, c='g',s=16)
    plt.plot(xfit, pred, '-b');
    plt.plot(xfit, target_function(xfit), '-k', alpha=0.7,
            linestyle = 'dashed');
    plt.title(model_name + ' Fit ')
    f.savefig(GitDir + model_name + ' Fit.pdf',bbox_inches = "tight")
    
def mean_pred(pred):
    '''
    Args:
        pred - a matrix of size (num_samples, num_points) of predictions
        
    Return: a (1, num_points) array of means of columns
    '''
    return np.mean(pred, axis=0)

def var_pred(pred):
     '''
     Args:
        pred - a matrix of size (num_samples, num_points) of predictions
        
     Return: a (1, num_points) array of means of columns
     '''
     return np.var(pred, axis=0)

def bias(xfit,pred):
    '''
    Args: as in mean_pred
    '''
    return target_function(xfit) - mean_pred(pred)

def bias_squared_graphs(xfit, dt_regressors, gb_regressors):
    dt_pred = make_predictions(dt_regressors,xfit)
    gb_pred = make_predictions(gb_regressors,xfit)
    f = plt.figure(figsize=[16,8])
    plt.plot(xfit, bias(xfit,dt_pred)**2, '-b', label='Decision Tree');
    plt.plot(xfit, bias(xfit,gb_pred)**2, '-k', alpha=0.7,
            linestyle = 'dashed', label='Boosting');
    plt.legend(loc='best')
    plt.title( ' Bias Comparison ')
    f.savefig(GitDir + 'Bias Comparison.pdf',bbox_inches = "tight")
    
def variance_graphs(xfit,dt_regressors, gb_regressors):
    dt_pred = make_predictions(dt_regressors,xfit)
    gb_pred = make_predictions(gb_regressors,xfit)
    f = plt.figure(figsize=[16,8])
    plt.plot(xfit, var_pred(dt_pred), '-b', label='Decision Tree');
    plt.plot(xfit, var_pred(gb_pred), '-k', alpha=0.7,
            linestyle = 'dashed', label='Boosting');
    plt.legend(loc='best')
    plt.title( ' Variance Comparison ')
    f.savefig(GitDir + 'Variance Comparison.pdf',bbox_inches = "tight")
    
def squared_error_graphs(xfit,dt_regressors, gb_regressors):
    '''
    Graphs bias**2 + var
    '''
    dt_pred = make_predictions(dt_regressors,xfit)
    gb_pred = make_predictions(gb_regressors,xfit)
    f = plt.figure(figsize=[16,8])
    plt.plot(xfit, bias(xfit,dt_pred)**2 + var_pred(dt_pred), '-b', label='Decision Tree');
    plt.plot(xfit, bias(xfit,gb_pred)**2 + var_pred(gb_pred), '-k', alpha=0.7,
            linestyle = 'dashed', label='Boosting');
    plt.legend(loc='best')
    plt.title( ' Squared Error Comparison ')
    f.savefig(GitDir + 'Squared Error Comparison.pdf',bbox_inches = "tight")
    
def squared_error_integral(regressors,xfit):
    '''
    First produce a matrix of predictions, one row for each sample, one column
    for each x in sfit.
    err(sample,x) =  target - prediction
    mn is empirical mean (expectation) of err**2 over samples
    Then return average over x
    
    This function integrates a squared_error_graph wrt x.
    '''
    pred = make_predictions(regressors,xfit)
    err = target_function(xfit) - pred
    mn = np.mean(err**2,axis=0) # mean over samples
    return np.mean(mn)  # mean over xfit

def test_error(regressor,x_train, y_train, x_test,y_test,depth):
    tree = regressor(x_train,y_train,depth)
    pred = make_predictions([tree],x_test)
    error = y_test - pred
    return np.mean(error**2)
    
def main():  
    num_features = 200
    len_interval = 10.0
    num_samples = 100
    num_points = 1000
    xfit = np.linspace(0, len_interval, num_points)
    decision_tree_depth = 9
    sigma = 0.2
    
    X_data, Y_data = generate_data_samples(num_samples,
                                           len_interval, 
                                           num_features, 
                                           sigma) 
    
    dt_regressors = regressor_list(decision_tree_regressor_one_sample,
                               X_data,
                               Y_data,
                               decision_tree_depth)

    gb_regressors = regressor_list(GradientBoostingRegressor_one_sample,
                               X_data,
                               Y_data,
                               decision_tree_depth)
    ind = 4
    graph_data(X_data[ind], Y_data[ind], xfit)
    graph(X_data[ind], Y_data[ind], xfit,dt_regressors[ind],'Decision Tree')
    graph(X_data[ind], Y_data[ind], xfit,gb_regressors[ind],'SKLearn Boosting')
    
    bias_squared_graphs(xfit,dt_regressors,gb_regressors)
    variance_graphs(xfit,dt_regressors,gb_regressors)
    squared_error_graphs(xfit,dt_regressors,gb_regressors)
    
    print('dt_L2',np.round(squared_error_integral(dt_regressors,xfit),4))
    print('gb_L2',np.round(squared_error_integral(gb_regressors,xfit),4))
    
    ind_train = 4
    ind_test = 10

    dt_error = test_error(decision_tree_regressor_one_sample,
                          X_data[ind_train],
                          Y_data[ind_train],
                          X_data[ind_test],
                          Y_data[ind_test],
                          decision_tree_depth)

    gb_error = test_error(GradientBoostingRegressor_one_sample,
                          X_data[ind_train],
                          Y_data[ind_train],
                          X_data[ind_test],
                          Y_data[ind_test],
                          decision_tree_depth)
    
    print('dt_test_error',np.round(dt_error,3))
    print('gb_test_error',np.round(gb_error,3))

    
if __name__ == "__main__":
    main()