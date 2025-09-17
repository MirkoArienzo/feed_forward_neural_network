import numpy as np
import utils



class MLPClassifier:
    """
    L-layer fully-connected neural network classifier with L2 regularization,
    optimizers (gd, momentum, adam)
    """
    
    def __init__(self,
                 layers_dims,
                 activations=None,
                 initialization='he',
                 optimizer='adam',
                 learning_rate=1e-3,
                 lambd=0.0,
                 beta=0.9,
                 beta1=0.9,
                 beta2=0.0999,
                 epsilon=1e-8,
                 seed=1):
        np.random.seed(seed)
        self.layers_dims = list(layers_dims)
        self.L = len(self.layers_dims) - 1 # number of layers
        
        if self.L < 1:
            raise ValueError("layers_dims must contain at least two elements [input, output]")
            
        if activations is None:
            activations = []
            for l in range(1, self.L - 1):
                activations.append('relu')
            activations.append('sigmoid' if self.layers_dims[-1] == 1 else 'softmax')
        self.activations = activations
        
        self.initialization = initialization
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lambd = float(lambd)
        self.seed = seed
        
        # Optimizer hyperparameters
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Internal state
        self.parameters = {}
        self.v = {} # momentum
        self.s = {} # adam second moment
        self.t = 0 # adam timestep
        self._initialization_parameters()
        self._initialization_optimizer_state()
        
        
        
    # Initialization
    def _initialization_parameters(self):
        np.random.seed(self.seed)
        params = {}
        for l in range(1, self.L + 1):
            # print(l)
            # print(self.layers_dims[l-1])
            n_prev = self.layers_dims[l-1]
            n_l = self.layers_dims[l]
            
            if self.initialization == 'he' and (self.activations[l-1].lower() == 'relu'):
                scale = np.sqrt(2 / n_prev)
            elif self.initialization == 'xavier':
                scale = np.sqrt(1 / n_prev)
            else:
                scale = 0.01
            params['W' + str(l)] = np.random.randn(n_l, n_prev) * scale
            params['b' + str(l)] = np.zeros((n_l, 1))
        self.parameters = params
            
    def _initialization_optimizer_state(self):
        self.v = {}
        self.s = {}
        for l in range(1, self.L + 1):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            self.v['dW' + str(l)] = np.zeros_like(W)
            self.v['db' + str(l)] = np.zeros_like(b)
            self.s['dW' + str(l)] = np.zeros_like(W)
            self.s['db' + str(l)] = np.zeros_like(b)
        self.t = 0
    
    # Forward propagation
    def _linear_forward(self, A_prev, W, b):

        # print(f"W: {W.shape}")        
        # print(f"A_prev: {A_prev.shape}")
        # print(f"b: {b.shape}")
        Z = W @ A_prev + b
        cache = (A_prev, W, b, Z)
        return Z, cache
    
    def _apply_activation(self, Z, activation):
        act = activation.lower()
        if act == 'relu':
            return utils.relu(Z)
        elif act == 'sigmoid':
            return utils.sigmoid(Z)
        elif act == 'tanh':
            return utils.tanh(Z)
        elif act == 'softmax':
            return utils.softmax(Z)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
    def forward(self, X, training=True):
        
        caches = []
        A = X
        
        for l in range(1, self.L + 1):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z, _ = self._linear_forward(A, W, b)
            activation = self.activations[l-1].lower()
        
            # final layer: sigmoid or softmax
            A_next = self._apply_activation(Z, activation)
        
            cache = {
                'A_prev': A, 
                'Z': Z,
                'W': W,
                'b': b,
                'activation': activation}
            caches.append(cache)
            A = A_next
                
        AL = A
        
        return AL, caches
    
    # Compute cross-entropy cost function
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        
        eps = 1e-12
        
        AL_clipped = np.clip(AL, eps, 1-eps)
        if self.layers_dims[-1] == 1:
            #binary cross-entropy
            cost = utils.binary_cross_entropy(AL_clipped, Y)
        else:
            #multiclass cross-entropy for softmax: Y assumed one-hot (n_y, m)
            cost = utils.multiclass_cross_entropy(AL_clipped, Y)
        
        # Apply L2 regularization
        if self.lambd is not None and self.lambd > 0:
            sum_squares = 0
            for l in range(1, self.L+1):
                W = self.parameters['W' + str(l)]
                sum_squares += np.sum(np.square(W))
            regularization_cost = self.lambd / (2*m) * sum_squares
        
        return cost + regularization_cost
    
    # Backward propagation
    def backward(self, AL, Y, caches):
        grads = {}
        m = Y.shape[1]
        
        dZ = AL - Y
        
        for l in reversed(range(1, self.L +1)):
            cache = caches[l-1]
            A_prev = cache['A_prev']
            W = cache['W']
            
            # Compute and save derivatives in grads dictionary
            dW = 1/m * dZ @ A_prev.T + 1/m * self.lambd * W
            db = 1/m * np.sum(dZ, axis=1, keepdims=True)
            
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db
            
            if l > 1:
                dA_prev = W.T @ dZ
                # now through activation of previous layer to get dZ for next iteration
                prev_activation = self.activations[l-2].lower()
                Z_prev = caches[l-2]['Z']
                if prev_activation == 'relu':
                    dZ = utils.relu_backward(dA_prev, Z_prev)
                elif prev_activation == 'sigmoid':
                    dZ = utils.sigmoid_backwards(dA_prev, Z_prev)
                elif prev_activation == 'tanh':
                    dZ = utils.tanh_backwards(dA_prev, Z_prev)
                elif prev_activation == 'softmax':
                    # rarely used for hidden layers; treat generically via derivative (not implemented)
                    raise ValueError("Softmax should only be used on the final layer.")
                else:
                    raise ValueError(f"Unsupported activation: {prev_activation}")
            else:
                # done
                pass
        return grads
    
    # Parameters updated using gd
    def _update_parameters_gd(self, grads, learning_rate):
        
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    
    # Parameters update using momentum
    def _update_parameters_momentum(self, grads, learning_rate, beta):
       
        for l in range(1, self.L + 1):
        
            self.v['dW' + str(l)] = beta * self.v['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
            self.v['db' + str(l)] = beta * self.v['db' + str(l)] + (1 - beta) * grads['db' + str(l)]
            self.parameters['W' + str(l)] -= learning_rate * self.v['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * self.v['db' + str(l)]
    
    # Parameters update using Adam
    def _update_parameters_adam(self, grads, learning_rate, beta1, beta2, epsilon):
       
        # update timestep
        self.t += 1
       
        for l in range(1, self.L + 1):
            
            # update biased first moment estimate
            self.v['dW' + str(l)] = beta1 * self.v['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
            self.v['db' + str(l)] = beta1 * self.v['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]
            
            # update biased second raw moment estimate
            self.s['dW' + str(l)] = beta2 * self.s['dW' + str(l)] + (1 - beta2) * (grads['dW' + str(l)] ** 2)
            self.s['db' + str(l)] = beta2 * self.s['db' + str(l)] + (1 - beta2) * (grads['db' + str(l)] ** 2)
            
            # compute bias-corrected estimates
            v_corrected_dW = self.v['dW' + str(l)] / (1 - beta1 ** self.t)
            v_corrected_db = self.v['db' + str(l)] / (1 - beta1 ** self.t)
            s_corrected_dW = self.s['dW' + str(l)] / (1 - beta2 ** self.t)
            s_corrected_db = self.s['db' + str(l)] / (1 - beta2 ** self.t)
            
            # update parameters
            self.parameters['W' + str(l)] -= learning_rate * v_corrected_dW / (np.sqrt(s_corrected_dW) + epsilon)
            self.parameters['b' + str(l)] -= learning_rate * v_corrected_db / (np.sqrt(s_corrected_db) + epsilon)
    
    
    ## Method for mini-batchs
    @staticmethod
    def _random_mini_batches(X, Y, mini_batch_size=64, seed=0):
        
        np.random.seed(seed)
        m = X.shape[1]
        
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        
        mini_batches = []
        num_complete = m // mini_batch_size
        
        for k in range(num_complete):
            mini_X = shuffled_X[:, k*mini_batch_size: (k+1)*mini_batch_size]
            mini_Y = shuffled_Y[:, k*mini_batch_size: (k+1)*mini_batch_size]
            mini_batches.append((mini_X, mini_Y))
        
        if m%mini_batch_size != 0:
            mini_X = shuffled_X[:, num_complete*mini_batch_size: m]
            mini_Y = shuffled_Y[:, num_complete*mini_batch_size: m]
            mini_batches.append((mini_X, mini_Y))
        
        return mini_batches
    
    ## Fit and train the model
    def fit(self, X_train, Y_train, X_test=None, Y_test=None,
            num_epochs=1000, mini_batch_size=64, learning_rate=None,
            print_cost=True, seed=0):
        """
        Train the network:
            X: (n_x, m)
            Y: (n_y, m) for multiclass or (1, m) for binary
            X_test, Y_test: optional test set
        Returns history dictionary
        """
        
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        m = X_train.shape[1]
        history = {
            'train_costs': [],
            'test_costs': []
            }
        best_test_cost = np.inf
        best_parames = None
        seed_local = seed
        
        for epoch in range(num_epochs):
            
            seed_local += 1
            minibatches = self._random_mini_batches(X_train, Y_train, mini_batch_size, seed_local)
            cost_total = 0
            
            for minibatch in minibatches:
                [minibatch_X, minibatch_Y] = minibatch
                AL, caches = self.forward(minibatch_X, training=True)
                cost = self.compute_cost(AL, minibatch_Y)
                cost_total += cost * minibatch_X.shape[1]
                grads = self.backward(AL, minibatch_Y, caches)
                
                #update
                if self.optimizer == 'gd':
                    self._update_parameters_gd(grads, learning_rate)
                elif self.optimizer == 'momentum':
                    self._update_parameters_momentum(grads, learning_rate, self.beta)
                elif self.optimizer == 'adam':
                    self._update_parameters_adam(grads, learning_rate, self.beta1,
                                                     self.beta2, self.epsilon)
                else:
                    raise ValueError("Unsupported optimizer: " + self.optimizer)
            
            # epoch finished
            
            avg_cost = cost_total / m
            history['train_costs'].append(avg_cost)
            
            # test cost
            if X_test is not None and Y_test is not None:
                AL_test, _ = self.forward(X_test, training=False)
                test_cost = self.compute_cost(AL_test, Y_test)
                history['test_costs'].append(test_cost)
            else:
                test_cost = None
            
            if print_cost and (epoch % max(1, num_epochs // 10) == 0):
                if test_cost is None:
                    print(f"Epoch {epoch}/{num_epochs} - train_cost: {avg_cost:.6f}")
                else:
                    print(f"Epoch {epoch}/{num_epochs} - train_cost: {avg_cost:.6f} - test_cost: {test_cost:.6f}")   
                
            
        return history
    
    def predict_probabilities(self, X):
        AL, _ = self.forward(X, training=False)
        return AL
    
    def predict(self, X, threshold=0.5):
        AL = self.predict_probabilities(X)
        if self.layers_dims[-1] == 1:
            return (AL > threshold).astype(int)
        else:
            # multiclass: argmax
            return np.argmax(AL, axis=0).reshape(1, -1)
    
    def accuracy(self, X, Y):
        predictions = self.predict(X)
        if self.layers_dims[-1] == 1:
            truth = Y.astype(int)
        return np.mean(predictions.flatten() == truth)
            
            
            
            
            
    
