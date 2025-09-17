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
            
 

            
