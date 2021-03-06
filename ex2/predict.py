#from numpy import round
import numpy as np

from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters.
#               You should set p to a vector of 0's and 1's
    z = np.dot(X, theta)
    theta.shape = (3,1)
    
    p = np.copy(z)
    p = (sigmoid(p))
    
    p = np.where(p > 0.5, 1, 0)
    #p = np.where(p <= 0.5, 0, 1)
   
        

# =========================================================================

 #   p = 0
    return p