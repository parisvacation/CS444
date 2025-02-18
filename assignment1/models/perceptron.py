"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = X_train.shape
        # L2 Regularization weight
        lambda_reg = 0.005
        # Margin used for multi-class classification, but does it become specific SVM model? 
        margin = 5.0

        # Binary Classification
        if self.n_class == 2:
            # First, pre-process the data
            X_train_used = np.hstack([X_train, np.ones((N, 1))])  # Add bias term, the shape is (N, D+1)
            y_train_used = np.where(y_train == 0, -1, 1) # Replace 0 with -1, the shape is (N,)

            # Second, initialize w
            if self.w is None:
                np.random.seed(42)
                self.w = np.random.uniform(-1, 1, size=(D + 1,)) * 0.01 # the shape is (D + 1,)

            # Third, train the model
            for epoch in range(self.epochs):
                # Shuffle the training data
                indices = np.random.permutation(N)
                X_train_new = X_train_used[indices]
                y_train_new = y_train_used[indices]

                for i in range(N):
                    # Use perceptron update rule
                    if y_train_new[i] * (np.dot(X_train_new[i], self.w)) < 0:
                        self.w += self.lr * y_train_new[i] * X_train_new[i]

                # Update the learning8 rate (Learing rate decay)
                self.lr *= 0.95
        

        # Multi-Class Classification
        else:
            # First, pre-process the data
            X_train_used = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
            X_train_used = np.hstack([X_train_used, np.ones((N, 1))])
            y_train_used = y_train

            # Second, initialize w
            if self.w is None:
                np.random.seed(100)
                self.w = np.random.uniform(-1, 1, size=(self.n_class, D + 1)) * 0.01 # the shape is (n_class, D + 1)

            # Third, train the model
            for epoch in range(self.epochs):
                # Shuffle the training data
                indices = np.random.permutation(N)
                X_train_new = X_train_used[indices]
                y_train_new = y_train_used[indices]

                for i in range(N):
                    # Use perceptron update rule for multi-class classification
                    scores = np.dot(self.w, X_train_new[i])
                    # L2 Regularization
                    self.w *= (1 - self.lr * lambda_reg / N)
                    # Update w if wrong class score is higher than the correct class score
                    for k in range(self.n_class):
                        if k != y_train_new[i] and scores[k] > scores[y_train_new[i]] - margin:
                            self.w[y_train_new[i]] += self.lr * X_train_new[i]
                            self.w[k] -= self.lr * X_train_new[i]

                # Update the learning rate (Learing rate decay)
                self.lr *= 0.80

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

        # Binary classification
        if self.n_class == 2:
            # Add bias term
            X_test_used = np.hstack([X_test, np.ones((N, 1))])
            # Use the trained weights to predict labels for test data points
            y_pred = np.where(np.dot(X_test_used, self.w) >= 0, 1, 0)
            return y_pred
        
        # Multi-class classification
        else:
            # Standardlize and add bias term
            X_test_used = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
            X_test_used = np.hstack([X_test_used, np.ones((N, 1))])
            # Use the trained weights to predict labels for test data points
            y_pred = np.argmax(np.dot(X_test_used, self.w.T), axis=1)
            return y_pred
        
