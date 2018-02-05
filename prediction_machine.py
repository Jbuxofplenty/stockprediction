import pickle
import hashlib
import MySQLdb
import datetime
import argparse
import glob, os
import numpy as np
import pandas as pd
from datetime import timedelta
from model_db import ModelDatabase
from mlp_regression import Numbers
from mlp_regression import MLPRegressor
from feature_vector import FeatureVectorizor
from linear_regression import LinearRegressor
from sklearn.model_selection import train_test_split

class Numbers:
    """
    Class to load and store the generated feature vector
    """
    def __init__(self, fname_X=None, fname_Y=None):
        if fname_X is None:
            fname_X = "pickled_files/training_data/sd_X.pkl"
        if fname_Y is None:
            fname_Y = "pickled_files/training_data/sd_Y.pkl"
        self.X_dict, self.params = pickle.load(open(fname_X, 'rb'))
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

    def dump_X(self, fname='sd_X', serial_num=None):
        with open('pickled_files/training_data/' + fname + '_' + str(serial_num) + '_' + str(self.params['table']) + '.pkl', 'wb') as f:
            pickle.dump([self.X_dict, self.params], f)

    def dump_Y(self, fname='sd_Y', serial_num=None):
        with open('pickled_files/training_data/' + fname + '_' + str(serial_num) + '_' + str(self.params['table']) + '.pkl', 'wb') as f:
            pickle.dump(self.Y_dict, f)

class PredictionMachine:
    """
    Class to load and store a database filled with predictions and actual prices
    """
    def __init__(self, stock):
        self.max_day_pred = 280
        self.start_date = str(datetime.date.today())
        self.end_date = str(datetime.date.today() + timedelta(days=self.max_day_pred))
        self.str_rng = []
        self.rng = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.df = pd.DataFrame(columns=['actual'], index=self.rng)
        self.table = stock
        self.model_db = None
        self.X = None
        self.cur_model = None
        self.cur_prices = {}

        # store the str range for indexing
        for date in self.rng:
            self.str_rng.append(date.strftime('%m-%d-%Y'))
    """
    Function to automatically update all of the training files for the models
    """
    def update_models(self, keys=None):
        if keys is None:
            keys = self.model_db.keys()
        for key in keys:
            serial_num = key
            model_params = self.model_db[key]['model_params']
            X_params = self.model_db[key]['X_params']

            # Number of data points
            fv = FeatureVectorizor(params=X_params)
            fname = 'sd_X_' + str(key)
            fv.load_X(fname=fname)
            fname = 'sd_Y_' + str(key)
            fv.load_Y(fname=fname)
            # Cycle through the number of days at the given step size to make X and Y
            find_value = True
            fv.start_date = datetime.date.today()
            while(find_value):
                try:
                    fv.X[str(fv.start_date)]
                    find_value = False
                except:
                    find_value = True
                    feature_vector, output = fv.gen_feature_vector()
                    fv.start_date -= timedelta(days=1)

            # Store the X and Y vectors into files
            fv.dump_X()
            fv.dump_Y()

            # Open the new updated pickle file for training data
            fname_X = "pickled_files/training_data/sd_X_" + X_params['table'] + ".pkl"
            fname_Y = "pickled_files/training_data/sd_Y_" + X_params['table'] + ".pkl"
            data = Numbers(fname_X=fname_X, fname_Y=fname_Y)

            # Make a new MLP Regressor with the optimal parameters
            if self.model_db[key]['type'] == 'Linear Regressor':
                lr = LinearRegressor(train_x=data.train_x, train_y=data.train_y, test_x=data.test_x, test_y=data.test_y, params=model_params)
                lr.train()
            elif self.model_db[key]['type'] == 'MLP Regressor':
                mlpr = MLPRegressor(train_x=data.train_x, train_y=data.train_y, test_x=data.test_x, test_y=data.test_y, params=model_params)
                mlpr.train()

            # Store the model with the appended serial_number
            if self.model_db[key]['type'] == 'Linear Regressor':
                fname_model = "pickled_files/models/lr_regression_" + str(serial_num) + "_" + X_params['table'] + ".pkl"
                lr.dump(fname_model)
            elif self.model_db[key]['type'] == 'MLP Regressor':
                fname_model = "pickled_files/models/mlp_regression_" + str(serial_num) + "_" + X_params['table'] + ".pkl"
                mlpr.dump(fname_model)

            # Store the data that trained the model
            data.dump_X(serial_num=serial_num)
            data.dump_Y(serial_num=serial_num)

    """
    Function to cycle through each of the models and predict
    """
    def predict_models(self, keys=None):
        if keys is None:
            keys = self.model_db.keys()
        for key in keys:
            if self.model_db[key]['X_params']['table'] == self.table:
                if self.model_db[key]['type'] == 'MLP Regressor':
                    fname = "pickled_files/models/mlp_regression_" + str(key) + "_" + self.table + ".pkl"
                    model = LinearRegressor()
                elif self.model_db[key]['type'] == 'Linear Regressor':
                    fname = "pickled_files/models/lr_regression_" + str(key) + "_" + self.table + ".pkl"
                    model = MLPRegressor()
                self.cur_model = model.load(fname)
                self.X = self.load_X(key)
                self.predict_prices(key)
                # Check to make sure the
                if not str(key) in self.df.columns:
                    self.df[str(key)] = pd.Series(self.cur_prices)
                for price_key in self.cur_prices.keys():
                    if price_key in self.df[str(key)].index:
                        if pd.isnull(self.df[str(key)][price_key]):
                            self.df[str(key)][price_key] = self.cur_prices[price_key]
                    else:

                        self.df = self.df.reindex(pd.to_datetime(self.df.index.union(self.rng)))
                        self.df[str(key)][price_key] = self.cur_prices[price_key]
                mask = (self.df.index > pd.to_datetime(datetime.date.today() - timedelta(days=30)))
                print(self.df.loc[mask])

    """
    Function to store a model's prices specified by the serial number
    """
    def predict_prices(self, serial_num):
        days_out_prediction = self.model_db[serial_num]['num_days']
        end_date = str(datetime.date.today() + timedelta(days=days_out_prediction))
        pred_rng = pd.date_range(start=str(datetime.date.today()), end=end_date, freq='D')
        today = datetime.date.today()
        self.cur_prices = {}
        for i, day in enumerate(pred_rng):
            if i >= len(pred_rng) - 2:
                continue
            day_feat_vec = day - days_out_prediction
            day_formatted = pd.to_datetime(str(day_feat_vec))
            feat_vec = self.X[str(day_formatted).split()[0]]
            prediction = self.cur_model.predict(np.reshape(feat_vec, (1, -1)))
            self.cur_prices[pd.to_datetime(day)] = prediction[0]

    """
    Function to load X which contains the feature vectors
    """
    def load_X(self, serial_num):
        self.X, params = pickle.load(open("pickled_files/training_data/sd_X_" + str(serial_num) + "_" + self.table + ".pkl", 'rb'))
        return self.X

    """
    Function to load the model database
    """
    def load_model_db(self, filename=None):
        if filename == None:
            self.model_db = pickle.load(open("pickled_files/model_dbs/model_db.pkl", 'rb'))
            return self.model_db
        else:
            self.model_db = pickle.load(open(filename, 'rb'))
            return self.model_db

    """
    Function to update all of the actual prices
    """
    def update_actual(self):
        # Retrieve all of the prices and timestamps from the db
        query = 'SELECT timestamp, price FROM ' + self.table + ';'
        data = self.db_retrieve_data(query)

        # Format the data
        formatted_data = {}
        for tup in data:
            db_time, price = tup
            if db_time is not None and price is not None:
                db_time = pd.to_datetime(str(db_time))
                price = float(price)
                formatted_data[db_time] = price

        # Update the pandas dataframe
        for db_time in formatted_data.keys():
            if not db_time in self.df.index:
                self.df = self.df.reindex(pd.to_datetime(self.df.index.union(self.rng)))
                #print(db_time, '\t', formatted_data[db_time])
            self.df.set_value(db_time, 'actual', formatted_data[db_time])

    """
    Function to retrieve data from the database and return it
    """
    def db_retrieve_data(self, query):
        db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                             user="josiah",         # your username
                             passwd="adversereaction",  # your password
                             db="stock_data")        # name of the data base
        cur = db.cursor()
        try:
            # Execute the SQL command
            cur.execute(query)
            data = cur.fetchall()
            # Commit your changes in the database
            db.commit()
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            # Rollback in case there is any error
            db.rollback()
            return None
        db.close()
        return data

    """
    Function to load a previously stored database
    """
    def load(self, filename=None):
        start_date = self.find_last_saved_date()
        if filename == None:
            self.df = pickle.load(open('pickled_files/prediction_machine/predictions_' + self.table + '_' + start_date + '.pkl', 'rb'))
            return self.df
        else:
            self.df = pickle.load(open(filename, 'rb'))
            return self.df

    """
    Function to load a previously stored database
    """
    def find_last_saved_date(self):
        date = datetime.date.today() - timedelta(days=1)
        start_date = str(date)
        found = False
        while(not found):
            date -= timedelta(days=1)
            start_date = str(date)
            for f in glob.glob("pickled_files/prediction_machine/*" + start_date + "*"):
                found = True
        return start_date

    """
    Function to store the database
    """
    def dump(self, filename=None):
        start_date = str(datetime.date.today())
        start_date.replace('/', '_')
        if filename == None:
            pickle.dump(self.df, open('pickled_files/prediction_machine/predictions_' + self.table + '_' + start_date + '.pkl', 'wb'))
        else:
            pickle.dump(self.df, open(filename, 'wb'))

    """
    Function to export the database to an excel file for further analysis
    """
    def export_excel(self, filename=None):
        start_date = str(datetime.date.today())
        start_date.replace('/', '_')
        if filename == None:
            writer = pd.ExcelWriter('analysis/' + self.table + '_predictions_' + start_date + '.xlsx')
        else:
            writer = pd.ExcelWriter(filename)
        new_df = self.format_excel()
        new_df.to_excel(writer, 'Predictions')
        writer.save()

    """
    Function to format the dataframe in order to print to excel
    """
    def format_excel(self):
        start_date = str(datetime.date.today() - datetime.timedelta(days=30))
        rng = pd.date_range(start=start_date, end=self.end_date, freq='D')
        new_df = pd.DataFrame(columns=[self.df.columns], index=rng)
        for date in rng:
            if date in self.df.index:
                new_df.loc[date,:] = self.df.loc[date,:]
        return new_df

    """
    Function to store a backup of the database
    """
    def backup_db(self, filename=None):
        start_date = str(datetime.date.today())
        start_date.replace('/', '_')
        if filename == None:
            pickle.dump(self.df, open('pickled_files/prediction_machine/' + self.table + '_predictions_' + start_date + '.pkl', 'wb'))
        else:
            pickle.dump(self.df, open(filename, 'wb'))

if __name__ == '__main__':
    # Load the symbols that are stored from the original stock_price_data.py script
    fname='pickled_files/misc/symbols'
    with open(fname + '.pkl', 'rb') as f:
        tables = pickle.load(f)
        for table in tables:
            pm = PredictionMachine(table)
            #pm.load()
            pm.load_model_db()
            pm.update_models()
            pm.update_actual()
            pm.predict_models()
            pm.dump()
            pm.export_excel()
        print(pm.model_db)
