# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:15:53 2019

@author: D2
"""

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from typing import Tuple, List
GitDir = "C:/Users/D2/Documents/Github/Regressors/"

rng = np.random.RandomState(2)

def target_function(x):
    fast_oscillation = 0.5*np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    return slow_oscillation + fast_oscillation

def generate_one_data_sample(len_interval, size_dataset, sigma=0.1):
    '''
    Args:
        len_interval - the x values are uniformly distributed in the interval
        [0, len_interval]
        size_dataset - the number of (x,y) values
        sigma - the standard deviation of the y-values
        
    Returns: a tuple consisting of two arrays:
        x is the np.array of num_features numbers
        y is the np.array of corresponding y values
        
        One data sample is one vector of x values and one vector of y values
    '''
    x = len_interval * rng.rand(size_dataset) #rand is Uniform[0, 1]
    x = np.sort(x)
    noise = sigma * rng.randn(len(x)) #randn is N(0,1)
    y = target_function(x) + noise
    return (x,y)
    
def generate_data_samples(
    num_datasets: int,
    len_interval: float, 
    size_dataset: int, 
    sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Args:
        num_datasets - the number of x-arrays and y-arrays produced by one 
        iteration of generate_one_data_sample
        len_interval - the x values are uniformly distributed in the interval
        [0, len_interval]
        size_dataset - the number of (x,y) values
        sigma - the standard deviation of the y-values
        
    Return: two matrices of shapes (num_datasets, size_dataset) 
    one whose rows are samples of x and the other whose
    rows are samples of y.
    The values of x can change from one dataset to the next.
    See Bishop, p. 148.
    '''
    data = [generate_one_data_sample(len_interval, size_dataset, sigma) 
            for i in range(num_datasets) ]
    x_list, y_list = list(zip(*data))
    return (np.array(x_list), np.array(y_list))
    

def decision_tree_regressor_one_sample(x, y, depth):
    '''
    Args:
        x, y - one data sample
        depth - the maximum depth the decision tree should go.
        
    Returns a regressor object fit to x, y
    The function graph uses regressor in
    pred = regressor.predict(xfit[:, None]) = a (num_points,) array
    '''
    dt = DecisionTreeRegressor(random_state=0, max_depth=depth)
    dt.fit(x[:, None], y)
    return dt
    
def GradientBoostingRegressor_one_sample(x, y, n_estimators=100):   
     '''
     See decision_tree_regressors
     '''
     gb = GradientBoostingRegressor(n_estimators=n_estimators, 
                                    learning_rate=1.0, 
                                    max_depth=1)
     gb.fit(x[:, None], y)
     return gb
 
def regressor_list(reg, X, Y, depth):
    '''
    Args:
        X, Y - two matrices, one row from each constitutes on data sample
        depth - as in decision_tree_regressor_one_sample
        
    Return: a list of regressors fitted to x. Each row 
    comes from one data sample.
    '''
    return [reg(x, y, depth) for x, y in zip(X,Y)]


    
def make_predictions(regressors, xfit):
    '''
    Inputs - a list of regressors and xfit
    
    Return: np.array converts the list of fitted regressors into a matrix
    with one row per sample
    '''
    return np.array([r.predict(xfit[:, None]) for r in regressors])

def graph_data(x, y, xfit):
    plt.figure(figsize=[16,8])
    plt.scatter(x,y, c='g',s=16)
    plt.plot(xfit, target_function(xfit), alpha=0.7,
            linestyle = 'dashed');
    plt.title('Target Function and a Data Set')
#    f.savefig(GitDir + 'Target Function and a Data Set.pdf',bbox_inches = "tight")

   
def graph(x, y, xfit, regressor, model_name, depth):
    pred = regressor.predict(xfit[:, None])
    print(pred.shape)
    plt.figure(figsize=[16,8])
    plt.scatter(x,y, c='g',s=16)
    plt.plot(xfit, pred, c='b');
    plt.plot(xfit, target_function(xfit), alpha=0.7,
            linestyle = 'dashed');
    plt.title(model_name + ' Fit with max depth = ' + str(depth))
#    f.savefig(GitDir + model_name + ' Fit.pdf',bbox_inches = "tight")
    
def graph_datasets(
    X: np.ndarray, 
    Y: np.ndarray, 
    row_nos: List[int], 
    xfit: np.ndarray, 
    regressors: list, 
    model_name:str, 
    complexity_name: str, 
    complexity_param: int):
    """
    X, Y are the full datasets
    X.shape = (num_datasets, num_datapoint) = Y.shape
    indices select with datasets to plot
    """
    plt.figure(figsize = [16, 24])
#    plt.figure(figsize = (16, 8))
    num_plots = len(row_nos)

    sum_of_regressors = np.zeros_like(xfit)
   
    for index in range(num_plots):
        ax = plt.subplot(num_plots+1, 1, index + 1)
#        
        row = row_nos[index]
        ax.scatter(X[row], Y[row], c='g', s=16)
        ax.plot(xfit, regressors[row].predict(xfit[:, None]), '-b')
        ax.plot(xfit, target_function(xfit), c='k', alpha=0.7,
                linestyle="dashed")
        ax.set_title(model_name + ' Fit ' + str(row) + ' with ' + complexity_name + '  = ' + str(complexity_param))
        
        sum_of_regressors += regressors[row].predict(xfit[:, None])

    ax = plt.subplot(num_plots+1, 1, num_plots+1)
    ax.plot(xfit, sum_of_regressors/num_plots, '-b')
    ax.plot(xfit, target_function(xfit), 'k', alpha=0.7,
                linestyle="dashed")
    ax.set_title(model_name + ' average fit  with ' + complexity_name + '  = ' + str(complexity_param))

def mean_pred(pred):
    '''
    Args:
        pred - a matrix of size (num_datasets, num_points) of predictions
        
    Return: a (1, num_points) array of means of columns (average over data sets)
    '''
    return np.mean(pred, axis=0)

def var_pred(pred):
     '''
     Args:
        pred - a matrix of size (num_datasets, num_points) of predictions
        
     Return: a (1, num_points) array of vars of columns
     '''
     return np.var(pred, axis=0)

def bias(xfit, pred):
    '''
    Args: as in mean_pred
    '''
    return target_function(xfit) - mean_pred(pred)

def bias_graphs(xfit, dt_regressors, gb_regressors):
    dt_pred = make_predictions(dt_regressors, xfit)
    gb_pred = make_predictions(gb_regressors, xfit)
    plt.figure(figsize=[16,8])
    plt.plot(xfit, bias(xfit, dt_pred)**1, c='b', label='Decision Tree');
    plt.plot(xfit, bias(xfit, gb_pred)**1, c='k', alpha=0.7,
            linestyle = 'dashed', label='Boosting');
    plt.legend(loc='best')
    plt.title( ' Bias Comparison')
#    f.savefig(GitDir + 'Bias Comparison.pdf',bbox_inches = "tight")
    
def bias_v2(xfit, regressors, regressor_names):
    """
    regressors is a tuple of regressor lists to compare
    regressor_names is a list of names to put in the plt labels
    """
    preds = [make_predictions(regs, xfit) for regs in regressors]
    plt.figure(figsize=[16, 8])
    plt.plot(xfit, bias(xfit, preds[0])**1, c='b', label=regressor_names[0]);
    plt.plot(xfit, bias(xfit, preds[1])**1, color ='black', alpha=0.7,
            linestyle = 'dotted', label=regressor_names[1]);
    plt.legend(loc='best')
    plt.title( ' Bias Comparison')
    
def variance_graphs(xfit,dt_regressors, gb_regressors):
    dt_pred = make_predictions(dt_regressors, xfit)
    gb_pred = make_predictions(gb_regressors, xfit)
    plt.figure(figsize=[16,8])
    plt.plot(xfit, var_pred(dt_pred), color='b', label='Decision Tree');
    plt.plot(xfit, var_pred(gb_pred), color ='black', alpha=0.7,
            linestyle = 'dotted', label='Boosting');
    plt.legend(loc='best')
    plt.title( ' Variance Comparison ')
#    f.savefig(GitDir + 'Variance Comparison.pdf',bbox_inches = "tight")
    
def variance_v2(xfit, regressors, regressor_names):
    preds = [make_predictions(regs, xfit) for regs in regressors]
    plt.figure(figsize=[16, 8])
    plt.plot(xfit, var_pred(preds[0]), c='b', label=regressor_names[0]);
    plt.plot(xfit, var_pred(preds[1]), c='k', alpha=0.7,
            linestyle = 'dashed', label=regressor_names[1]);
    plt.legend(loc='best')
    plt.title( ' Variance Comparison ')

    
def squared_error_graphs(xfit, dt_regressors, gb_regressors, depth):
    '''
    Graphs bias**2 + var
    '''
    dt_pred = make_predictions(dt_regressors,xfit)
    gb_pred = make_predictions(gb_regressors,xfit)
    plt.figure(figsize=[16,8])
    plt.plot(xfit, bias(xfit,dt_pred)**2 + var_pred(dt_pred), c='b', label='Decision Tree');
    plt.plot(xfit, bias(xfit,gb_pred)**2 + var_pred(gb_pred), c='k', alpha=0.7,
            linestyle = 'dashed', label='Boosting');
    plt.legend(loc='best')
    plt.title( ' Squared Error Comparison with max depth = '+ str(depth))
#    f.savefig(GitDir + 'Squared Error Comparison.pdf',bbox_inches = "tight")
    
def squared_error_integral(regressors, xfit):
    '''
    First produce a matrix of predictions, one row for each sample, one column
    for each x in sfit.
    err(sample,x) =  target - prediction
    mn is empirical mean (expectation) of err**2 over samples
    Then return average over x
    
    This function integrates a squared_error_graph wrt x.
    '''
    pred = make_predictions(regressors, xfit)
    err = target_function(xfit) - pred
    mn = np.mean(err**2, axis=0) # mean over datasets
    return np.mean(mn)  # mean over xfit

def test_error(regressor, x_train, y_train, x_test, y_test, complexity_param):
    reg = regressor(x_train, y_train, complexity_param)
    pred = make_predictions([reg], x_test)
    error = y_test - pred
    return np.mean(error**2)
    
def main(
    num_datapoints: int = 200,  # size of one data set
    sigma: float = 0.3,
    num_datasets: int = 100,   # multiple data sets for bias-variance tradeoff
    len_interval: float = 10.0,
    num_points: int = 1000,
    decision_tree_depth: int = 9, 
    num_gb_estimators: int=512,
    num_graphs: int = 4,
):  
    X_data, Y_data = generate_data_samples(
       num_datasets,
       len_interval, 
       num_datapoints, 
       sigma
    ) 
    print(f"Data size: {X_data.shape[0]} datasets, \
    {X_data.shape[1]} data points per dataset")
    
    dt_regressors = [decision_tree_regressor_one_sample(x, y,
                                        decision_tree_depth)
                                        for x, y in zip(X_data, Y_data)]

    gb_regressors = [GradientBoostingRegressor_one_sample(x, y, 
                                        n_estimators = num_gb_estimators
                                        )
                                        for x, y in zip(X_data, Y_data)]

    indices = range(0,num_graphs+1,1)
    xfit = np.linspace(0, len_interval, num_points)
    graph_datasets(
        X_data, 
        Y_data, 
        indices, 
        xfit, 
        dt_regressors, 
        'Decision Tree', 
        'depth', 
        decision_tree_depth
    )
    graph_datasets(
        X_data,
        Y_data, 
        indices, 
        xfit, 
        gb_regressors, 
        'SKLearn Boosting',
        'num_estimators', 
        num_gb_estimators
    )
    
  

def bias_variance(complexity):
    """
    complexity is a list of numbers of gb_estimators; 
    currently len(complexity) must == 2.
    function plots bias and variance of GBRegressors with different numbers of
    estimators
    
    The less complex GBM generally does not track the oscillations in the true
    function. Upon averaging over many training sets, the mean is biased. But the
    variance is lower because the low complexity GBMs are more rigid.
    
    The greater complexity GBMs can follow the sinusoidal oscillations.
    Hence the average of many such from different training datasets also
    tracks the sinusoid => lower bias.
    But each complex GBM fits the data more tightly, so the variance of the more
    complex GBMs is higher..
    """
    num_datapoints = 200  # size of one data set
    num_datasets = 100   # multiple data sets for bias-variance tradeoff
    
    len_interval = 10.0
    num_points = 1000
    xfit = np.linspace(0, len_interval, num_points)
    
    sigma = 0.3
    X_data, Y_data = generate_data_samples(num_datasets,
                                           len_interval, 
                                           num_datapoints, 
                                           sigma) 
    gb_regressors = {num_gb_estimators : [GradientBoostingRegressor_one_sample(x, y, 
                                        n_estimators = num_gb_estimators)
                                        for x, y in zip(X_data, Y_data)]
                                        for num_gb_estimators in complexity}
#    bias_graphs(xfit, gb_regressors[complexity[0]], gb_regressors[complexity[1]])
    bias_v2(xfit,
            [gb_regressors[c] for c in complexity],
            ["Gradient boosting. n_estimators = " + str(c) for c in complexity])
#    variance_graphs(xfit, gb_regressors[complexity[0]], gb_regressors[complexity[1]])
    variance_v2(xfit,
            [gb_regressors[c] for c in complexity],
            ["Gradient boosting. n_estimators = " + str(c) for c in complexity])
    
if __name__ == "__main__":
    
#    X_data, Y_data = generate_data_samples(
#        num_datasets=5,
#        len_interval=10, 
#        size_dataset=8, 
#        sigma=0.3
#    ) 
#    print(np.round(X_data, 2))
#    print(np.round(Y_data, 2))
    main(
        num_datapoints = 20,  # size of one data set
        sigma =  0.3,
        num_datasets = 100,   # multiple data sets for bias-variance tradeoff
        len_interval = 10.0,
        num_points = 256,
        decision_tree_depth = 2,
        num_gb_estimators = 512,
        num_graphs=4
    )
#    main(
#        num_datapoints = 200,  # size of one data set
#        sigma =  0.3,
#        num_datasets = 100,   # multiple data sets for bias-variance tradeoff
#        len_interval = 10.0,
#        num_points = 1000,
#        decision_tree_depth = 2,
#        num_gb_estimators = 512,
#        num_graphs=4
#    )
    """
    complexities = [32, 2**14]
    bias_variance(complexities)
    """
