import gzip
import math
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from model_db import ModelDatabase
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
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
        X_dict, params = pickle.load(open(fname_X, 'rb'))
        Y_dict = pickle.load(open(fname_Y, 'rb'))
        self.X = []
        self.Y = []

        for key_X, key_Y in zip(X_dict.keys(), Y_dict.keys()):
            self.X.append(X_dict[key_X])
            self.Y.append(Y_dict[key_Y])

        training_data, test_data, training_labels, test_labels = train_test_split(self.X, self.Y, test_size=0.2, shuffle=False)

        self.train_x = training_data
        self.train_y = training_labels
        self.test_x = test_data
        self.test_y = test_labels
        self.params = params

    def dump_X(self, fname='sd_X', serial_num=None):
        with open('pickled_files/training_data/' + fname + '_' + str(serial_num) + '.pkl', 'wb') as f:
            pickle.dump([self.X, self.params], f)

    def dump_Y(self, fname='sd_Y', serial_num=None):
        with open('pickled_files/training_data/' + fname + '_' + str(serial_num) + '.pkl', 'wb') as f:
            pickle.dump(self.Y, f)

class LinearRegressor:
    '''
    Linear Regression classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1):
        '''
        initialize Linear Regression classifier
        '''
        # Store the training and test data
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        # Store the parameters for the model
        self.params = {}
        self.params['fit_intercept'] = fit_intercept
        self.params['normalize'] = normalize
        self.params['copy_X'] = copy_X
        self.params['n_jobs'] = n_jobs

        # Create the model
        self.model = linear_model.LinearRegression(fit_intercept=self.params['fit_intercept'], normalize=self.params['normalize'], copy_X=self.params['copy_X'], n_jobs=self.params['n_jobs'])

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
            return pickle.load(open("pickled_files/linear_regression.pkl", 'rb'))
        else:
            return pickle.load(open(filename, 'rb'))

    def dump(self, filename=None):
        if filename == None:
            pickle.dump(self.model, open("pickled_files/linear_regression.pkl", 'wb'))
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
    lr_best_params = {'fit_intercept':True, 'normalize':False, 'copy_X':True, 'n_jobs':1}
    lr = LinearRegressor(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    lr.train()

    # Store the trained model into the database
    model_db = ModelDatabase()
    model_db.load()
    model_db.store_cur_data([data.params['days_out_prediction'], 'Linear Regressor'], columns=['num_days', 'type'])
    model_db.store_cur_data([data.params, lr.params, len(data.train_x), len(data.test_x), len(data.train_x[0])], columns=['X_params', 'model_params', 'num_train', 'num_test', 'num_features'])
    hash_X = model_db.find_hash(fname_X)
    model_db.store_cur_data([hash_X], columns=['X_hash'])
    serial_num = model_db.find_serial_number()

    # Store the model with the appended serial_number
    fname_model = "pickled_files/models/linear_regression_" + str(serial_num) + ".pkl"
    lr.dump(fname_model)
    hash_model = model_db.find_hash(fname_model)
    model_db.store_cur_data([hash_model], columns=['model_hash'])

    # Store the data that trained the model
    data.dump_X(serial_num=serial_num)
    data.dump_Y(serial_num=serial_num)

    # Store all the data in the data db and dump the db
    model_db.store_data(serial_num)
    model_db.dump()

    # Analyze the model
    lr_acc = lr.evaluate()
    print(lr_acc)

    # Make predictions using the testing set
    pred_y = lr.predict(data.test_x)

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
