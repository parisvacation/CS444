"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.train_mean = None
        self.train_std = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.

        #return -1/(1+np.exp(z))+1
        return np.where(z<=-7,-1/(1+np.exp(z))+1,1/(np.exp(-z)+1))


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01.
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # Preprocess the data
        self.train_mean = np.mean(X_train,axis=0)
        self.train_std = np.std(X_train,axis=0)
        X_train = (X_train-self.train_mean) / self.train_std

        self.w=0.0004*np.random.uniform(-25,25,size=(np.shape(X_train)[1],))

        turn_counter=0
        decay_it=1

        while turn_counter<self.epochs:
            train_value=X_train@self.w
            train_log=self.sigmoid(train_value)

            decay_it=1*decay_it
            turn_counter=1+turn_counter

            diff=(train_log-y_train)
            gradi=(X_train.T)@diff/np.shape(X_train)[0]

            self.w=self.w-gradi*decay_it*self.lr



    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        # Preprocess the data
        X_test = (X_test-self.train_mean) / self.train_std

        counting=0
        predi_result=np.full(np.shape(X_test)[0],53)

        trained_w=self.w
        predi_val=X_test@trained_w
        predi_log=self.sigmoid(predi_val)


        while np.shape(X_test)[0]>counting:
            if predi_log[counting]<self.threshold:
                predi_result[counting]=0
            else:
                predi_result[counting]=1
            counting=1+counting

        return predi_result
