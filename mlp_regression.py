import gzip
import math
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network
from model_db import ModelDatabase
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
            fname_X = "pickled_files/training_data/sd_X.pkl"
        if fname_Y is None:
            fname_Y = "pickled_files/training_data/sd_Y.pkl"
        self.X_dict, params = pickle.load(open(fname_X, 'rb'))
        self.Y_dict = pickle.load(open(fname_Y, 'rb'))
        self.X = []
        self.Y = []

        for key_X, key_Y in zip(self.X_dict.keys(), self.Y_dict.keys()):
            self.X.append(self.X_dict[key_X])
            self.Y.append(self.Y_dict[key_Y])

        training_data, test_data, training_labels, test_labels = train_test_split(self.X, self.Y, test_size=0.2, shuffle=False)

        self.train_x = training_data
        self.train_y = training_labels
        self.test_x = test_data
        self.test_y = test_labels
        self.params = params

    def dump_X(self, fname='sd_X', serial_num=None):
        with open('pickled_files/training_data/' + fname + '_' + str(serial_num) + '.pkl', 'wb') as f:
            pickle.dump([self.X_dict, self.params], f)

    def dump_Y(self, fname='sd_Y', serial_num=None):
        with open('pickled_files/training_data/' + fname + '_' + str(serial_num) + '.pkl', 'wb') as f:
            pickle.dump(self.Y_dict, f)

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

        # Store the parameters for the model
        self.params = {}
        self.params['hidden_layer_sizes'] = hidden_layer_sizes
        self.params['activation'] = activation
        self.params['solver'] = solver
        self.params['alpha'] = alpha
        self.params['batch_size'] = batch_size
        self.params['learning_rate'] = learning_rate
        self.params['learning_rate_init'] = learning_rate_init
        self.params['power_t'] = power_t
        self.params['max_iter'] = max_iter
        self.params['shuffle'] = shuffle
        self.params['random_state'] = random_state
        self.params['tol'] = tol
        self.params['verbose'] = verbose
        self.params['warm_start'] = warm_start
        self.params['momentum'] = momentum
        self.params['nesterovs_momentum'] = nesterovs_momentum
        self.params['early_stopping'] = early_stopping
        self.params['validation_fraction'] = validation_fraction
        self.params['beta_1'] = beta_1
        self.params['beta_2'] = beta_2
        self.params['epsilon'] = epsilon

        # Create the model
        self.model = neural_network.MLPRegressor(hidden_layer_sizes=self.params['hidden_layer_sizes'], activation=self.params['activation'],
            solver=self.params['solver'], alpha=self.params['alpha'], batch_size=self.params['batch_size'], learning_rate=self.params['learning_rate'],
            learning_rate_init=self.params['learning_rate_init'], power_t=self.params['power_t'], max_iter=self.params['max_iter'], shuffle=self.params['shuffle'],
            random_state=self.params['random_state'], tol=self.params['tol'], verbose=self.params['verbose'], warm_start=self.params['warm_start'],
            momentum=self.params['momentum'], nesterovs_momentum=self.params['nesterovs_momentum'], early_stopping=self.params['early_stopping'],
            validation_fraction=self.params['validation_fraction'], beta_1=self.params['beta_1'], beta_2=self.params['beta_2'], epsilon=self.params['epsilon'])

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
            return pickle.load(open("pickled_files/models/mlp_regression.pkl", 'rb'))
        else:
            return pickle.load(open(filename, 'rb'))

    def dump(self, filename=None):
        if filename == None:
            pickle.dump(self.model, open("pickled_files/models/mlp_regression.pkl", 'wb'))
        else:
            pickle.dump(self.model, open(filename, 'wb'))

    def predict(self, x):
        """
        evaluates the prediction for a given X
        """
        return self.model.predict(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression Classifier Options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    fname_X = "pickled_files/training_data/sd_X.pkl"
    fname_Y = "pickled_files/training_data/sd_Y.pkl"
    data = Numbers(fname_X=fname_X, fname_Y=fname_Y)

    # Perform cross validation on each of the optimal models and show the accuracy
    mlpr = MLPRegressor(train_x=data.train_x[:args.limit], train_y=data.train_y[:args.limit], test_x=data.test_x, test_y=data.test_y)
    mlpr.train()

    # Store the trained model into the database
    model_db = ModelDatabase()
    model_db.load()
    model_db.store_cur_data([data.params['days_out_prediction'], 'MLP Regressor'], columns=['num_days', 'type'])
    model_db.store_cur_data([data.params, mlpr.params, len(data.train_x), len(data.test_x), len(data.train_x[0])], columns=['X_params', 'model_params', 'num_train', 'num_test', 'num_features'])
    hash_X = model_db.find_hash(fname_X)
    model_db.store_cur_data([hash_X], columns=['X_hash'])
    serial_num = model_db.find_serial_number()

    # Store the model with the appended serial_number
    fname_model = "pickled_files/models/mlp_regression_" + str(serial_num) + ".pkl"
    mlpr.dump(fname_model)
    hash_model = model_db.find_hash(fname_model)
    model_db.store_cur_data([hash_model], columns=['model_hash'])

    # Store the data that trained the model
    data.dump_X(serial_num=serial_num)
    data.dump_Y(serial_num=serial_num)

    # Store all the data in the data db and dump the db
    model_db.store_data(serial_num)
    model_db.dump()

    # Analyze the model
    mlpr_acc = mlpr.evaluate()
    print(mlpr_acc)

    # Make predictions using the testing set
    pred_y = mlpr.predict(data.test_x)

    # The coefficients
    # print('Coefficients: \n', mlpr.model.coefs_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(data.test_y, pred_y))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(data.test_y, pred_y))

    # Plot outputs
    x = []
    for i, feat_vec in enumerate(data.test_x):
        x.append(i)
    plt.scatter(x, data.test_y,  color='black')
    plt.plot(x, pred_y, color='blue', linewidth=3)
    plt.show()
