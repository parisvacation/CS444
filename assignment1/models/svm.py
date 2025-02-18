"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
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

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        N = X_train.shape[0]

        if self.w is None:
            raise ValueError("w is not initialized")
        grad_w = np.zeros(self.w.shape)

        # Binary classification
        if self.n_class == 2:
            X_batch = X_train
            y_batch = np.where(y_train == 0, -1, 1)

            # First, calculate the gradient of regularization term(without bias)
            grad_w[:-1] += self.reg_const * self.w[:-1]

            # Second, calculate the gradient of data loss
            margin = y_batch * (X_batch @ self.w)
            mask = margin < 1
            grad_w -= (X_batch[mask].T @ y_batch[mask]) / N


        # Multi-class classification
        else:
            X_batch = X_train
            y_batch = y_train

            # First, calculate the gradient of regularization term(without bias)
            grad_w[:, :-1] += self.reg_const * self.w[:, :-1]

            # Second, calculate the gradient of data loss            
            scores = X_batch @ self.w.T # shape is (N, C)
            correct_scores = scores[np.arange(N), y_batch] # shape is (N,)

            margin = correct_scores[:, np.newaxis] - scores # shape is (N, C)
            margin_mask = margin < 1
            margin_mask[np.arange(N), y_batch] = False
            sum_mask = np.sum(margin_mask, axis=1) # shape is (N,)

            np.add.at(grad_w, y_batch, -(sum_mask[:, np.newaxis] * X_batch) / N)
            grad_w += (margin_mask.T @ X_batch) / N

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
        # Define the size of mini-batch
        batch_size = 9000

        # Binary classification
        if self.n_class == 2:
            # Mean-centering
            self.train_mean = np.mean(X_train, axis=0)
            X_train_used = X_train - self.train_mean
            # Add bias term
            X_train_used = np.hstack([X_train_used, np.ones((N, 1))])
            y_train_used = y_train
            
            # initialze w
            if self.w is None:
                np.random.seed(42)
                self.w = np.random.uniform(-1, 1, (D + 1, )) * 0.01

            for epoch in range(self.epochs):
                indices = np.random.permutation(N)
                X_train_used = X_train_used[indices]
                y_train_used = y_train_used[indices]

                # Loop through the training data
                for start_idx in range(0, N, batch_size):
                    end_idx = min(start_idx + batch_size, N)
                    batch_indices = indices[start_idx:end_idx]
                    X_train_new = X_train_used[batch_indices]
                    y_train_new = y_train_used[batch_indices]
                
                    # Calculate the gradient and update w
                    grad_w = self.calc_gradient(X_train_new, y_train_new)
                    self.w = self.w - self.lr * grad_w
                
                # Update the learning rate
                self.lr *= 0.95
        
        # Multi-class classification
        else:
            # Add bias term (Fashion-mnist has been mean-centered)
            X_train_used = np.hstack([X_train, np.ones((N, 1))])
            y_train_used = y_train

            # initialze w
            if self.w is None:
                np.random.seed(42)
                self.w = np.random.uniform(-1, 1, (self.n_class, D + 1)) * 0.01

            for epoch in range(self.epochs):
                indices = np.random.permutation(N)
                X_train_used = X_train_used[indices]
                y_train_used = y_train_used[indices]

                # Loop through the training data
                for start_idx in range(0, N, batch_size):
                    end_idx = min(start_idx + batch_size, N)
                    batch_indices = indices[start_idx:end_idx]
                    X_train_new = X_train_used[batch_indices]
                    y_train_new = y_train_used[batch_indices]
                
                    # Calculate the gradient and update w
                    grad_w = self.calc_gradient(X_train_new, y_train_new)
                    self.w = self.w - self.lr * grad_w
                
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

        # Binary classification
        if self.n_class == 2:
            # Mean-centering
            X_test_used = X_test - self.train_mean
            # Add bias term
            X_test_used = np.hstack([X_test_used, np.ones((N, 1))])
            y_pred = np.where(np.dot(X_test_used, self.w) >= 0, 1, 0)
        
        # Multi-class classification
        else:
            # Add bias term
            X_test_used = np.hstack([X_test, np.ones((N, 1))])
            y_pred = np.argmax(np.dot(X_test_used, self.w.T), axis=1)

        return y_pred
