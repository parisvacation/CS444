"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.train_mean = None
        self.train_std = None

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        """Calculate softmax scores.

        Parameters:
            scores: a numpy array of shape (N, C) where each column contains
                the scores for all possible classes.
        
        Returns:
            softmax scores for each example. Each row should sum to 1.

        """
        # Minus the max to avoid overflow
        Temperature = 1
        scores_exp = np.exp((scores - np.max(scores, axis=1, keepdims=True)) / Temperature)
        scores_result = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
        return scores_result
        
    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        N = y_train.shape[0]

        if self.w is None:
            ValueError("w is not initialized")
        grad_w = np.zeros_like(self.w)

        X_batch = X_train
        y_batch = y_train

        # First, calculate the gradient of regularization term(without bias)
        grad_w[:, :-1] += self.reg_const * self.w[:, :-1]

        # Second, calculate the gradient of the data loss
        scores = self.softmax(np.dot(X_batch, self.w.T))
        scores[range(N), y_batch] -= 1
        grad = scores / N
        grad_w += np.dot(grad.T, X_batch)

        return grad_w

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.
        
        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = X_train.shape
        # Determine the size of mini-batches 
        batch_size = 4096

        # Preprocess the data
        # Standardize and add bias term
        self.train_mean = np.mean(X_train, axis=0)
        self.train_std = np.std(X_train, axis=0)
        X_train_used = (X_train - self.train_mean) / self.train_std
        X_train_used = np.hstack((X_train_used, np.ones((N, 1))))
        y_train_used = y_train

        # Initialize w
        if self.w is None:
            np.random.seed(42)
            self.w = np.random.uniform(-1, 1, size=(self.n_class, D + 1)) * 0.01
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(N)
            X_train_used = X_train_used[indices]
            y_train_used = y_train_used[indices]

            # Loop through the whole training set
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                batch_indices = indices[start_idx:end_idx]
                X_train_new = X_train_used[batch_indices]
                y_train_new = y_train_used[batch_indices]

                # Calculate gradient and update w
                grad_w = self.calc_gradient(X_train_new, y_train_new)
                self.w -= self.lr * grad_w

            # Update the learning rate
            self.lr *= 0.95

        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        N = X_test.shape[0]
        
        # Standardize and add bias term
        X_test_used = (X_test - self.train_mean) / self.train_std
        X_test_used = np.hstack([X_test_used, np.ones((N, 1))])

        y_pred = np.argmax(np.dot(X_test_used, self.w.T), axis=1)

        return y_pred
