"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            opt: option for using "SGD" or "Adam" optimizer (Adam is Extra Credit)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        self.bn_eps = [1e-5] * (num_layers+1)
        self.cache = [None] * (num_layers+1)
        self.bn_gamma = [None] * (num_layers+1)
        self.bn_beta = [None] * (num_layers+1)
        self.dgamma = [None] * (num_layers+1)
        self.dbeta = [None] * (num_layers+1)
        
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            # Xavier initialization
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            
            # Batch normalization parameters
            self.bn_gamma[i] = np.ones(sizes[i])
            self.bn_beta[i] = np.zeros(sizes[i])
            self.dgamma[i] = np.zeros(sizes[i])
            self.dbeta[i] = np.zeros(sizes[i])
            
        # TODO: (Extra Credit) You may set parameters for Adam optimizer here
        if self.opt == "Adam":
            # Initialize the first and second moment
            self.m = {}
            self.v = {}
            for i in range(1, num_layers + 1):
                self.m["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
                self.v["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
                self.m["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])
                self.v["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])
                self.m["bn_gamma" + str(i)] = np.zeros_like(self.bn_gamma[i])
                self.v["bn_gamma" + str(i)] = np.zeros_like(self.bn_gamma[i])
                self.m["bn_beta" + str(i)] = np.zeros_like(self.bn_beta[i])
                self.v["bn_beta" + str(i)] = np.zeros_like(self.bn_beta[i])
            # Initialize the time step
            self.t = 0


    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        z = np.dot(X, W) + b
        return z
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray) -> np.ndarray:
        """Gradient of linear layer
        Parameters:
            W: the weight matrix
            X: the input data
            de_dz: the gradient of loss
        Returns:
            de_dw, de_db, de_dx
            where
                de_dw: gradient of loss with respect to W
                de_db: gradient of loss with respect to b
                de_dx: gradient of loss with respect to X
        """
        # TODO: implement me
        # The shape of de_dz is (N, C)
        N = de_dz.shape[0]

        # Calculate the gradient of total loss with respect to W, b, and X
        de_dw = np.dot(X.T, de_dz) # shape is (D, C)
        # de_db will be the summation of de_dz along the first dimension
        # de_db = np.dot(np.ones(N), de_dz)
        de_db = np.sum(de_dz, axis=0) # shape is (C,)
        de_dx = np.dot(de_dz, W.T) # shape is (N, D)
        return de_dw, de_db, de_dx

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.where(X > 0, X, 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
         # TODO: implement me
        return np.where(X > 0, 1, 0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return np.where(x > 0, 1/(1 + np.exp(-x)), 1 - 1/(1 + np.exp(x)))
    
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        # Assume y is the training value targets(ground truths)
        # Assume p is the training value outputs(predicted values)
        # Calculate the total loss (divided by y.shape[0]*y.shape[1])
        return np.mean((y - p) ** 2)
    
    def mse_separate(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # Calculate the loss for each output features (divided by y.shape[0])
        return np.mean((y - p) ** 2, axis=0)
     
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        # Calculate the gradient of total loss with respect to output p
        N, D = y.shape
        return -2 * (y - p) / (N * D)
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        # Calculate the gradient of total loss with respect to input x
        mse_grad_p = self.mse_grad(y, p)
        # According the sigmoid function value, calculate the gradient of sigmoid function
        mse_grad_x = mse_grad_p * (p * (1 - p))
        return mse_grad_x

    def bn_forward(self, X, idx):
        # Calculate the mean and variance of the input
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)
        std = np.sqrt(var + self.bn_eps[idx])

        # Normalize the input
        X_norm = (X - mu) / std

        # Scale and shift
        out = self.bn_gamma[idx] * X_norm + self.bn_beta[idx]

        # Store intermediate values for backward pass
        self.cache[idx] = (X, X_norm, mu, var, std)
        
        return out

    def bn_backward(self, dout, idx):
        
        # Retrieve intermediate values from the cache
        X, X_norm, mu, var ,std= self.cache[idx]
        N, D = X.shape

        # Calculate the gradients of beta and gamma
        self.dbeta[idx] = np.sum(dout, axis=0)
        
        self.dgamma[idx] = np.sum(dout * X_norm, axis=0)

        dX_norm = dout * self.bn_gamma[idx]
        
        # Calculate the gradients of X
        dX = (1.0 / N) * (1.0 / std) * (N * dX_norm - 
                                        np.sum(dX_norm, axis=0) - 
                                        X_norm * np.sum(dX_norm * X_norm, axis=0))
        return dX

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        # Store the input to do back-propagation later
        # print(X.shape)
        self.inputs = {}
        self.inputs["x"] = X

        # Store the output of each layer
        for idx in range(1, self.num_layers + 1):
            # Linear layer
            if idx == 1:
                self.outputs["z" + str(idx)] = self.linear(self.params["W" + str(idx)], X, self.params["b" + str(idx)])
            else:
                self.outputs["z" + str(idx)] = self.linear(self.params["W" + str(idx)], self.outputs["g" + str(idx - 1)], self.params["b" + str(idx)])
            
                
            if idx != self.num_layers:
                # Batch normalization
                self.outputs["bn" + str(idx)] = self.bn_forward(self.outputs["z" + str(idx)], idx)

            # Activation function
            if idx != self.num_layers:
                self.outputs["g" + str(idx)] = self.relu(self.outputs["bn" + str(idx)])
            else:
                self.outputs["g" + str(idx)] = self.sigmoid(self.outputs["z" + str(idx)])

        # (Extra Credit) Adam optimizer here

        # g[self.num_layers] is the final ouput.
        return self.outputs["g" + str(self.num_layers)] 

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        # Calculate the MSE loss
        MSE_loss = self.mse(y, self.outputs["g" + str(self.num_layers)])
        MSE_loss = float(MSE_loss)
        # print(y.shape)
        
        # Calculate the gradient of the MSE loss with respect to the final output
        self.gradients["de_dg" + str(self.num_layers)] = self.mse_grad(y, self.outputs["g" + str(self.num_layers)])

        for idx in range(self.num_layers, 0, -1):
            # Calculate the gradient of the MSE loss with respect to the last z, use element-wise multiplication
            if idx == self.num_layers:
                self.gradients["de_dz" + str(idx)] = self.gradients["de_dg" + str(idx)] * self.sigmoid_grad(self.outputs["z" + str(idx)])
            # Calculate the gradient of the MSE loss with respect to the other z, use element-wise multiplication
            else:
                d_activation = self.gradients["de_dg" + str(idx)] * self.relu_grad(self.outputs["bn" + str(idx)])
            
            if idx != self.num_layers:
                self.gradients["de_dz" + str(idx)] = self.bn_backward(d_activation, idx)
            
            # Calculate de_dw, de_db, de_dx/de_dg
            if idx == 1:
                de_dw, de_db, de_dx = self.linear_grad(
                    self.params["W" + str(idx)],
                    self.inputs["x"],
                    self.gradients["de_dz" + str(idx)]
                )
                self.gradients["x"] = de_dx
                self.gradients["W" + str(idx)] = de_dw
                # Turn de_db from 2D array to 1D array
                self.gradients["b" + str(idx)] = de_db
            
            else:
                de_dw, de_db, de_dx = self.linear_grad(
                    self.params["W" + str(idx)],
                    self.outputs["g" + str(idx - 1)],
                    self.gradients["de_dz" + str(idx)]
                )
                self.gradients["de_dg" + str(idx - 1)] = de_dx
                self.gradients["W" + str(idx)] = de_dw
                # Turn de_db from 2D array to 1D array
                self.gradients["b" + str(idx)] = de_db
         
        return MSE_loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        if self.opt == 'SGD':
            # TODO: implement SGD optimizer here
            for idx in range(1, self.num_layers + 1):
                self.params["W" + str(idx)] -= lr * self.gradients["W" + str(idx)]
                self.params["b" + str(idx)] -= lr * self.gradients["b" + str(idx)]
                self.bn_gamma[idx] -= lr * self.dgamma[idx]
                self.bn_beta[idx] -= lr * self.dbeta[idx]

        elif self.opt == 'Adam':
            # TODO: (Extra credit) implement Adam optimizer here
            # Increment the time step
            self.t += 1

            for idx in range(1, self.num_layers + 1):
                # Calculate the first moment for weight parameters
                self.m["W" + str(idx)] = b1 * self.m["W" + str(idx)] + (1 - b1) * self.gradients["W" + str(idx)]
                # Calculate the first moment for bias parameters
                self.m["b" + str(idx)] = b1 * self.m["b" + str(idx)] + (1 - b1) * self.gradients["b" + str(idx)]
                # Calculate the second moment for weight parameters
                self.v["W" + str(idx)] = b2 * self.v["W" + str(idx)] + (1 - b2) * np.square(self.gradients["W" + str(idx)])
                # Calculate the second moment for bias parameters
                self.v["b" + str(idx)] = b2 * self.v["b" + str(idx)] + (1 - b2) * np.square(self.gradients["b" + str(idx)])

                # Update the parameters using the Adam update rule
                m_w_hat = self.m["W" + str(idx)] / (1 - b1 ** self.t)
                v_w_hat = self.v["W" + str(idx)] / (1 - b2 ** self.t)
                m_b_hat = self.m["b" + str(idx)] / (1 - b1 ** self.t)
                v_b_hat = self.v["b" + str(idx)] / (1 - b2 ** self.t)
                self.params["W" + str(idx)] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
                self.params["b" + str(idx)] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
                
                # update the parameters for batch normalization
                if idx != self.num_layers:
                    # Calculate the first moment for gamma parameters
                    self.m["bn_gamma" + str(idx)] = b1 * self.m["bn_gamma" + str(idx)] + (1 - b1) * self.dgamma[idx]
                    # Calculate the first moment for beta parameters
                    self.m["bn_beta" + str(idx)] = b1 * self.m["bn_beta" + str(idx)] + (1 - b1) * self.dbeta[idx]
                    # Calculate the second moment for gamma parameters
                    self.v["bn_gamma" + str(idx)] = b2 * self.v["bn_gamma" + str(idx)] + (1 - b2) * np.square(self.dgamma[idx])
                    # Calculate the second moment for beta parameters
                    self.v["bn_beta" + str(idx)] = b2 * self.v["bn_beta" + str(idx)] + (1 - b2) * np.square(self.dbeta[idx])

                    # Update the parameters using the Adam update rule
                    m_gamma_hat = self.m["bn_gamma" + str(idx)] / (1 - b1 ** self.t)
                    v_gamma_hat = self.v["bn_gamma" + str(idx)] / (1 - b2 ** self.t)
                    m_beta_hat = self.m["bn_beta" + str(idx)] / (1 - b1 ** self.t)
                    v_beta_hat = self.v["bn_beta" + str(idx)] / (1 - b2 ** self.t)
                    self.bn_gamma[idx] -= lr * m_gamma_hat / (np.sqrt(v_gamma_hat) + eps)
                    self.bn_beta[idx] -= lr * m_beta_hat / (np.sqrt(v_beta_hat) + eps)
                
        else:
            raise NotImplementedError
        