import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        
        y[y==0]=-1
        X=np.insert(X,0,1,1)
        w=np.insert(w,0,b,0)
        
        for i in range(max_iterations+1):
            wX=np.dot(X,w.transpose())
            wX[wX==0]=-1
            ywX=y*wX
            
            X_loss=np.delete(X,np.where(ywX>0),0)
            y_loss=np.delete(y,np.where((ywX)>0))
            
            avg_loss=np.dot(X_loss.transpose(),y_loss)/N
            delta=step_size*avg_loss
            w=np.add(w,delta)
        
        b=w[0]
        w=w[1:]
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        
        y[y==0]=-1
        X=np.insert(X,0,1,1)
        w=np.insert(w,0,b,0)
        
        for i in range(max_iterations+1):
            wX=np.dot(X,w.transpose())
            prob_loss=sigmoid(-y*wX)
            
            avg_loss=np.dot(X.transpose(),y*prob_loss)/N
            delta=step_size*avg_loss
            w=np.add(w,delta)
        
        b=w[0]
        w=w[1:]            

        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1/(1+np.exp(-z))
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        X=np.insert(X,0,1,1)
        w=np.insert(w,0,b,0)
        
        y=np.dot(X,w.transpose())
        
        for i in range(len(y)):
            if y[i]>0:
                preds[i]=1
            else:
                preds[i]=0
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        X=np.insert(X,0,1,1)
        w=np.insert(w,0,b,0)
        
        y=np.dot(X,w.transpose())
        
        for i in range(len(y)):
            if sigmoid(y[i])>0.5:
                preds[i]=1
            else:
                preds[i]=0
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    
    
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w=np.zeros((C, D+1))
        b=np.zeros(C)
        X=np.insert(X,D,1,1)
        
        for i in range(max_iterations):
            
            n=np.random.choice(N)
            
            prob_numer=np.dot(X[n],w.transpose())
            prob_numer=np.exp(prob_numer-np.max(prob_numer))
            prob_denom=np.sum(prob_numer)
            g=np.divide(prob_numer, prob_denom)  
            g[y[n]]=g[y[n]]-1
            
            g=np.reshape(g,(C,1))
            Xn=np.reshape(X[n],(1,D+1))
            delta=step_size*np.dot(g, Xn)
            
            w = np.subtract(w,delta)
        
        b=w[:,D]
        w=np.delete(w,D,axis=1)
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D+1))
        b = np.zeros(C)
        X=np.insert(X,D,1,1)
        
        
        for i in range(max_iterations):
            prob_numer=np.dot(X,w.transpose())
            x_max=np.tile(np.amax(prob_numer,axis=0),(C,1))
            prob_numer=np.exp(np.subtract(prob_numer,x_max))
            prob_denom=np.tile(np.sum(prob_numer,axis=0),(C,1))
            prob=np.divide(prob_numer,prob_denom)
            prob[y,np.arange(N)]=prob[y,np.arange(N)]-1
            
            delta=step_size*np.dot(prob,X)/N
            
            w=np.subtract(w,delta)
        
        b=w[:,D]
        w=np.delete(w,D,axis=1)
            
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    
    X=np.insert(X,D,1,1)
    w=np.insert(w,D,b,1)
    y=np.dot(w,X.transpose())
    
    preds=np.argmax(y,axis=0)
    ############################################

    assert preds.shape == (N,)
    return preds