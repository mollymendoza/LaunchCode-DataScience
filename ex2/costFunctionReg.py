#from costFunction import costFunction
from sigmoid import sigmoid
import numpy as np
import pandas as pd



def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples
    J = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

    
    y = y.values
    z = np.dot(X, theta)
    hypothesis = sigmoid(z)
    yZero= np.dot(-y.T, np.log(hypothesis))
    yOne= np.dot((1-y).T, np.log(1-hypothesis))
    Jtemp = (np.sum(yZero-yOne))*(1/m)
    J = Jtemp + (Lambda/(2*m))*(np.sum(np.square(theta)))

    
   
    
    

# =============================================================
    #J = 0
    return J
