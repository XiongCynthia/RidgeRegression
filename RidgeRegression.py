import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted

class RidgeRegression:
    '''
    A scikit-learn-compatible class for fitting and predicting data on a Ridge regression model.
    '''
    def __init__(self, learning_rate=.01, L2_penalty=1, iterations=1000):
        self.lr = learning_rate # Learning rate for gradient descent
        self.l2 = L2_penalty # L2 penalty (lambda)
        self.iters = iterations # For gradient descent

    def fit(self, X, y):
        '''
        Fits data on a Ridge regression model.

        Args:
            X (numpy.array|torch.Tensor): Training data
            y (numpy.array|torch.Tensor): Target values
        '''
        # Convert NumPy arrays into PyTorch tensors
        if type(X) is np.ndarray:
            X = torch.from_numpy(X)
        self.X_ = X
        if type(y) is np.ndarray:
            y = torch.from_numpy(y)
        self.y_ = y
        
        if len(X.shape) == 1: # If X is a 1-D array
            self.W_ = torch.Tensor([0])
            self.X_ = X.reshape(-1,1)
        else:
            self.W_ = torch.zeros(X.shape[1])
        self.b_ = torch.tensor(0) # Coefficient

        # Gradient descent learning
        for iter in range(self.iters):
            self.__gradient_descent()
        return self
        
    def predict(self, X):
        '''
        Predict using the fitted Ridge regression model.

        Args:
            X (numpy.array|torch.Tensor): Sample data
        '''
        check_is_fitted(self) # fit() should have been called first
        if type(X) is np.ndarray:
            X = torch.from_numpy(X)
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        return X.matmul(self.W_.double()) + self.b_
    
    def __gradient_descent(self):
        '''
        Performs an iteration of gradient descent and updates weights.
        '''
        y_pred = self.predict(self.X_)
        
        m = self.X_.shape[0] # Number of observations (rows)
        dW = (-2 * self.X_.T.matmul(self.y_ - y_pred) + 
              2 * self.l2 * self.W_) / m # Gradients
        db = -2 * torch.sum(self.y_ - y_pred) / m # Coefficient
        
        # Update weights
        self.W_ = self.W_ - self.lr * dW
        self.b_ = self.b_ - self.lr * db
        return self
    
    def get_params(self):
        '''
        Returns parameters for the Ridge regression model in a dictionary.
        '''
        return {'learning_rate':self.lr, 'L2_penalty':self.l2, 'iterations':self.iters,
                'X_':self.X_, 'y_':self.y_, 'W_':self.W_, 'b_':self.b_}
