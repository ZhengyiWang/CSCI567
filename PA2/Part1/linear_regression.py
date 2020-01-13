"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y): #square and absoulte change
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    err = None
    
    if np.size(X,1)==np.size(w,0):
        d=np.dot(X,w)
    else:
        d=np.dot(X.transpose(),w)
    
    sub=np.abs(np.subtract(d,y))
    err=np.mean(sub,dtype=np.float64)
    
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  w = None
  
  XX=np.dot(X.transpose(),X)
  inv=np.linalg.inv(XX)
  Xy=np.dot(X.transpose(),y)
  w=np.dot(inv,Xy) 
  
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    w = None
    
    M=np.dot(X.transpose(),X)
    eigenValues,eigenVectors=np.linalg.eig(M)
    min_eValue=np.amin(np.absolute(eigenValues))
    I=0.1*np.identity(np.size(X,1))
    while min_eValue<0.00001:
        M=np.add(M,I)
        eigenValues,eigenVectors=np.linalg.eig(M)
        min_eValue=np.amin(np.absolute(eigenValues))
    
    Xy=np.dot(X.transpose(),y)
    inv=np.linalg.inv(M)
    w=np.dot(inv,Xy)
    
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    w = None
    
    M=np.dot(X.transpose(),X)
    I=lambd*np.identity(np.size(X,1))
    M=np.add(M,I)
    Xy=np.dot(X.transpose(),y)
    inv=np.linalg.inv(M)
    w=np.dot(inv,Xy)
    
    return w

###### Q1.5 ######  
#should be 1e^-14 something wrong 
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    
    min_error=float("inf")
    for i in range(-19, 20):
        w=regularized_linear_regression(Xtrain, ytrain,pow(10,i))
        current_error=mean_absolute_error(w, Xval, yval)
        if min_error>current_error:
            min_error=current_error
            bestlambda=pow(10,i)
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    T=X.copy()
    if power==1:
        return X
    for i in range(2,power+1):
        M=np.power(T,i)
        X=np.concatenate((X,M),axis=1)
    return X