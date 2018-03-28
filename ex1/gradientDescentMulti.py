from computeCostMulti import computeCostMulti
import numpy as np


def gradientDescentMulti(X, y, theta, alpha, num_iters):

    
    # Initialize some useful values
    C_history = []
    m = y.size  # number of training examples

    for i in range(0, num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
   
        theta = theta - alpha * (X.T.dot(X.dot(theta)-y)/m)
        
        
       
       
        
 
  
        # ============================================================
        # Save the cost J in every iteration

        C_history.append(computeCostMulti(X, y, theta))
    
    return theta, C_history
    