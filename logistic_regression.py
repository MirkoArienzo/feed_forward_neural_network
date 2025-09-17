import numpy as np
from utils import sigmoid
from typing import Optional, Dict, Any

class LogisticRegression:
    """
    Pure-NumPy logistic regression trained with (mini)batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Learning rate.
    num_iter : int
        Maximum number of gradient steps.
    l2 : float
        L2 regularization strength (0 disables).
    fit_intercept : bool
        Whether to learn an intercept term.
    batch_size : Optional[int]
        If None, uses full-batch GD; otherwise uses mini-batches.
    tol : Optional[float]
        Stop early if |J_{t-1} - J_t| < tol (after the first 10 iters).
    print_cost : bool
        Print loss every 100 steps.
    random_state : Optional[int]
        Seed for shuffling batches (if using mini-batch GD).
    """
    
    def __init__(self, learning_rate: float=0.5, num_iter: int =2000, l2: float = 0.0,
                  fit_intercept: bool = True, batch_size: Optional[int] = None,
                  tol: Optional[float] = None, print_cost: bool = False,
                  random_state: Optional[int] = None
                  ):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.batch_size = batch_size
        self.tol = tol
        self.print_cost = print_cost
        self.random_state = random_state
        
        # learned parameters
        self.w_: Optional[np.ndarray] = None # dimensions (n_features,)
        self.b_: float = 0.0
        self.costs_: list[float] = []
        
    def _linear(self, X:np.ndarray) ->np.ndarray:
        z = X @ self.w_
        if self.fit_intercept:
            z += self.b_
        return z

    def _propagate(self, X: np.ndarray, y:np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            dataset
        y : np.ndarray
            true labels
        Returns
        -------
        (loss, grad_w, grad_b)

        """
        
        m = X.shape[0]
        y_hat = sigmoid(self._linear(X))
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1-eps) # to avoid log(0)
        
        # compute loss as cross-entropy with L2 regularization
        cross_entropy_loss = - np.mean(y * np.log(y_hat) + (1-y) * np.log(1 - y_hat))
        reg_loss = 0.5 * self.l2 * np.sum(self.w_ ** 2) / m
        loss = cross_entropy_loss + reg_loss
        
        #compute gradients
        err = (y_hat - y) # shape (m,)
        grad_w = 1/m * X.T @ err + 1/m * self.l2 * self.w_
        grad_b = np.mean(err) if self.fit_intercept else 0.0
        
        return loss, grad_w, grad_b
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model

        Parameters
        ----------
        X : np.ndarray
            (m, n_features)
        y : np.ndarray
            (m,) or (m,1) with values in {0,1}
        """
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        m, n = X.shape
        
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
        else:
            rng = np.random.default_rng()

        # init
        self.w_ = np.zeros(n, dtype=float)
        self.b_ = 0.0
        self.costs_ = []
        
        batch_size = m if self.batch_size is None else int(self.batch_size)
        batch_size = max(1, min(batch_size, m))
        
        last_loss: Optional[float] = None
        
        for t in range(self.num_iter):
            # iterate over mini batches
            if batch_size < m:
                idx = rng.permutation(m)
                Xb_all, yb_all = X[idx], y[idx]
            else:
                Xb_all, yb_all = X, y
            
            for start in range(0, m, batch_size):
                stop = min(start + batch_size, m)
                Xb = Xb_all[start:stop]
                yb = yb_all[start:stop]
                
                loss, grad_w, grad_b = self._propagate(Xb, yb)
                self.w_ -= self.learning_rate * grad_w
                if self.fit_intercept:
                    self.b_ -= self.learning_rate * grad_b
            
            # track loss on full batch for logging/early stopping
            loss_full, _, _ = self._propagate(X, y)
            if t % 100 == 0:
                self.costs_.append(loss_full)
                if self.print_cost:
                    print(f"Cost after iteration {t}: {loss_full:.6f}")

            if self.tol is not None and last_loss is not None and t > 10:
                if abs(last_loss - loss_full) < self.tol:
                    break
            last_loss = loss_full

        return self
            
    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
            """
            Return P(y=1|x) as shape (m,).
            """
            X = np.asarray(X, dtype=float)
            return sigmoid(self._linear(X))            
         
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
            """
            Return class labels {0,1} as shape (m,).
            """
            return (self.predict_probabilities(X) >= threshold).astype(int)
        
    # scikit-learn-style plumbing
    def get_params(self) -> Dict[str, Any]:
        """
        Returns a dict of the estimator’s hyperparameters (the constructor args), e.g.
        """
        
        return {
            "learning_rate": self.learning_rate,
            "num_iter": self.num_iter,
            "l2": self.l2,
            "fit_intercept": self.fit_intercept,
            "batch_size": self.batch_size,
            "tol": self.tol,
            "verbose": self.print_cost,
            "random_state": self.random_state,
        }
    
    def set_params(self, **params) -> "LogisticRegression":
        """
        Sets hyperparameters from keyword args and returns self
        Enables
        model.set_params(lr=0.1, l2=1e-2).fit(X, y)
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self
            
    
"""
Usage example
# X: shape (m, n_features) ; y: shape (m,)
clf = LogisticRegressionGD(lr=0.5, num_iter=2000, l2=0.0, verbose=True).fit(X_train, y_train)
print("Train acc:", clf.score(X_train, y_train))
print("Test  acc:", clf.score(X_test, y_test))
probs = clf.predict_proba(X_test)
preds = clf.predict(X_test, threshold=0.5)
"""    

            


### OLD prototyping
# def initialize_with_zeros(dim):
#     """
#     Creates a vector of zeros of shape (dim,1) for w and initializes b to 0
    
#     Arguments:
#     dim : size of the w vector, int
    
#     Returns:
#     w : initialized vector of shape (dim, 1)
#     b : initialized scalar (bias)
#     """
    
#     return np.zeros((dim, 1)), 0

# def propagate(w, b, X, Y):
#     """
#     Implement cost function and its gradient for logistic regression

#     Arguments:
#     w : numpy array of size (x, 1)
#         vector of weights
#     b : scalar
#         bias
#     X : numpy array of size (x, num_examples)
#         dataset
#     Y : int 0/1
#         true "label" vector

#     Returns:
#     cost : cross-entropy cost function for logistic regression
#     dw : gradient of cost function
#     """
    
#     m = X.shape[1] # number of examples
    
#     Y_hat = sigmoid(np.dot(w.T, X) + b)
#     cost = -1/m * np.sum(Y * np.log(Y_hat) + (1-Y) * np.log(1 - Y_hat))
    
#     dw = 1/m * np.dot(X, (Y_hat - Y).T)
#     db = 1/m * np.sum(Y_hat - Y)
    
#     cost = np.squeeze(cost) # remove dimensions to make it a scalar
    
#     grads = {
#         "dw" : dw,
#         "db" : db
#         }
    
#     return grads, cost

# def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
#     """
#     Optimize w and b by running a gradient descent algorithm

#     Arguments:
#     w : weights, a numpy array of size (num_px * num_px * 3, 1)
#     b : bias, a scalar
#     X : data of shape (x, number of examples)
#     Y : true "label" of shape (1, number of examples)
#     num_iterations : number of iterations of the optimization loop
#     learning_rate : learning rate of the gradient descent update rule
#     print_cost : True to print the loss every 100 steps (for debugging)

#     Returns:
#     params : dictionary containing the weights w and bias b
#     grads : dictionary containing the gradients of the weights and bias
#     costs : list of all the costs computed during the optimization
#     """

#     costs = [] 
    
#     for i in range(num_iterations):
#         grads, cost = propagate(w, b, X, Y)
        
#         dw = grads["dw"]
#         db = grads["db"]
        
#         # update rule
#         w -= learning_rate * dw
#         b -= learning_rate * db

#         if i%100 == 0:
#             costs.append(cost)
#             if print_cost:
#                 print(f"Cost after iteration {i}: {cost}")
                
#     params = {
#         "w" : w,
#         "b" : b
#         }
    
#     grads = {
#         "dw" : dw,
#         "db" : db
#         }
    
#     return params, grads, costs

# def predict(w, b, X):
#     """
#     Predict whether the label is 0 or 1 using logistic regression parameters (w, b)

#     Arguments:
#     w : weights, a numpy array of size (x, 1)
#     b : bias, a scalar
#     X : data of size (x, number of examples)

#     Returns:
#     Y_prediction : a numpy array (vector) containing all predictions (0/1) for the examples in X
#     """
    
#     m = X.shape[1] # number of examples
#     Y_prediction = np.zeros((1, m))
    
#     Y_hat = sigmoid(np.dot(w.T, X) + b)
    
#     for i in range(Y_hat.shape[1]):
#         if Y_hat[0,i] > 0.5:
#             Y_prediction[0,i] = 1
    
#     return Y_prediction

# def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, 
#           learning_rate = 0.5, print_cost = False):
#     """
#     Builds the logistic regression model, includes both test and train

#     Arguments:
#     X_train : training set represented by a numpy array of shape (x, m_train)
#     Y_train : training labels represented by a numpy array (vector) of shape (1, m_train)
#     X_test : test set represented by a numpy array of shape (x, m_test)
#     Y_test . test labels represented by a numpy array (vector) of shape (1, m_test)
#     num_iterations : number of iterations to optimize the parameters
#     learning_rate : learning rate for gradient descent
#     print_cost : Set to true to print the cost every 100 iterations

#     Returns:
#     d : dictionary containing information about the model.
#     """
        
#     dim = np.shape(X_train.reshape[0],-1).T[1]
#     w, b = initialize_with_zeros(dim)
    
#     parameters, grads, costs = optimize(w, b, X_train, Y_train,
#                                         num_iterations, learning_rate, print_cost)
#     w = parameters["w"]
#     b = parameters["b"]

#     Y_prediction_train = predict(w, b, X_train)    
#     Y_prediction_test = predict(w, b, X_test)
    
    
#     accuracy_train = (100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100)
#     accuracy_test = (100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)
#     print(f"Train set accuracy: {accuracy_train}%")
#     print(f"Test set accuracy: {accuracy_test}%")
    
#     d = {"costs": costs,
#          "Y_prediction_test": Y_prediction_test,
#          "Y_prediction_train" : Y_prediction_train,
#          "w" : w,
#          "b" : b,
#          "learning_rate" : learning_rate,
#          "num_iterations": num_iterations}

#     return d
    
        
        
        
        
    

    