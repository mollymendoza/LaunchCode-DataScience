from numpy import asfortranarray, squeeze, asarray
import numpy as np
from gradientFunction import gradientFunction
from sigmoid import sigmoid

def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples
    grad = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
# 

#              derivatives of the cost w.r.t. each parameter in theta
   
    y = y.as_matrix().reshape((m, 1))
 
    #z = np.dot(X, theta)
    #hypothesis = sigmoid(z)
 
    grad = (gradientFunction(theta, X, y)  +  ((Lambda * theta) /m))
    grad[0] = (gradientFunction(theta, X, y))[0]
    


    
# =============================================================
    
    return grad