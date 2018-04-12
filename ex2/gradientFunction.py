from sigmoid import sigmoid
#from numpy import squeeze, asarray
import numpy as np


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    m = len(y)   # number of training examples
   # grad =  0
# =====#================= YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
    

       
    X = np.asarray(X)
    y = np.asarray(y)
    z = np.dot(X, theta)
    hypothesis = sigmoid(z)
    if y.shape != hypothesis.shape: y = y[:,0]
    

    grad = (1/m) * ((np.dot((hypothesis-y), X)))
    

# =============================================================
    
    return grad

#grad = gradientFunction(initial_theta, X, y)
