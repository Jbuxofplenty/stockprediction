import os
import gzip
import math
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class Numbers:
    """
    Class to load and store the generated feature vector
    """

    def __init__(self, fname_X=None, fname_Y=None):
        if fname_X is None:
            fname_X = "pickled_files/sd_X.pkl"
        if fname_Y is None:
            fname_Y = "pickled_files/sd_Y.pkl"
        X = pickle.load(open(fname_X, 'rb'))
        Y = pickle.load(open(fname_Y, 'rb'))

        training_data, test_data, training_labels, test_labels = train_test_split(X, Y, test_size=0.2, shuffle=False)

        self.train_x = training_data
        self.train_y = training_labels
        self.test_x = test_data
        self.test_y = test_labels

class MLPRegressor:
    '''
    MLP Regression classifier
    '''
    def __init__(self, train_x=None, train_y=None, test_x=None, test_y=None, hidden_layer_sizes=(100, ),
            activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant',
            learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001,
            verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        '''
        initialize MLP Regression classifier
        '''
        # Store the training and test data
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.feature_vector = []
        self.predictions = []
        self.predictions_strings = []
        self.predictions_fname = ''
        self.days_out_prediction = 0

        # Store the parameters for the model
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # Create the model
        self.model = neural_network.MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
            solver=self.solver, alpha=self.alpha, batch_size=self.batch_size, learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init, power_t=self.power_t, max_iter=self.max_iter, shuffle=self.shuffle,
            random_state=self.random_state, tol=self.tol, verbose=self.verbose, warm_start=self.warm_start,
            momentum=self.momentum, nesterovs_momentum=self.nesterovs_momentum, early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)

    def train(self):
        """
        trains the model with the training data passed to it
        """
        self.model.fit(self.train_x, self.train_y)

    def evaluate(self):
        """
        evaluates the accuracy of the training model
        """
        return self.model.score(self.test_x, self.test_y)

    def load(self, filename=None):
        if filename == None:
            self.model = pickle.load(open("pickled_files/mlp_regression.pkl", 'rb'))
            return self.model
        else:
            self.model = pickle.load(open(filename, 'rb'))
            return self.model

    def dump(self, filename=None):
        if filename == None:
            pickle.dump(self.model, open("pickled_files/mlp_regression.pkl", 'wb'))
        else:
            pickle.dump(self.model, open(filename, 'wb'))

    def predict(self, x):
        """
        evaluates the prediction for a given X
        """
        prediction = self.model.predict(x)
        self.predictions.append(prediction[0])
        return prediction

    def load_feature_vector(self, fname='sd_feature_vec'):
        with open('pickled_files/' + fname + '.pkl', 'rb') as f:
            self.feature_vector = pickle.load(f)

    def read_predictions_file(self, fname=None):
        # Use the latest file in the directory predictions to update
        if fname is None:
            last_modified = 0
            latest_time = 0
            for file in os.listdir("predictions/"):
                if file.endswith(".txt"):
                    predictions_fname = os.path.join("predictions/", file)
                    last_modified = os.path.getmtime(predictions_fname)
                    if latest_time <= last_modified:
                        self.predictions_fname = predictions_fname
                        latest_time = last_modified
        fname = self.predictions_fname
        with open(fname, 'r') as f:
            self.predictions_strings = f.readlines()
        self.days_out_prediction = len(self.predictions_strings)

    def write_predictions_file(self, fname=None):
        if fname is None:
            fname = self.predictions_fname
        with open(fname, 'w') as f:
            for string, pred in zip(self.predictions_strings, self.predictions):
                string = string.strip('\n')
                f.write(string + str(pred) + '\n')

if __name__ == '__main__':
    mlpr = MLPRegressor()
    mlpr.load()
    mlpr.read_predictions_file()
    for i in range(mlpr.days_out_prediction):
        mlpr.load_feature_vector(fname='sd_feature_vec_'+str(i))
        output = mlpr.predict(np.reshape(mlpr.feature_vector, (1, -1)))
        print(output[0])
    mlpr.write_predictions_file()
