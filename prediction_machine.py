import pickle
import hashlib
import MySQLdb
import datetime
import numpy as np
import pandas as pd
from datetime import timedelta
from model_db import ModelDatabase
from mlp_regression import MLPRegressor
from linear_regression import LinearRegressor

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

class PredictionMachine:
    """
    Class to load and store a database filled with predictions and actual prices
    """
    def __init__(self, stock):
        self.max_day_pred = 280
        self.start_date = str(datetime.date.today())
        self.end_date = str(datetime.date.today() + timedelta(days=self.max_day_pred))
        self.rng = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.df = pd.DataFrame(columns=['actual'], index=self.rng)
        self.table = stock
        self.model_db = None
        self.X = None
        self.cur_model = None
        self.cur_prices = {}

    """
    Function to cycle through each of the models and predict
    """
    def predict_models(self):
        for key in self.model_db.keys():
            if self.model_db[key]['type'] == 'MLP Regressor':
                fname = "pickled_files/models/mlp_regression_" + str(key) + ".pkl"
            elif self.model_db[key]['type'] == 'Linear Regressor':
                fname = "pickled_files/models/linear_regression_" + str(key) + ".pkl"
            model = MLPRegressor()
            self.cur_model = model.load(fname)
            self.X = self.load_X(key)
            self.predict_prices(key)
            self.df[str(key)] = pd.Series(self.cur_prices)

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
            print(day_feat_vec)
            day_formatted1 = pd.to_datetime(str(day_feat_vec).split()[0])
            day_formatted = datetime.datetime.strptime(str(day_feat_vec).split()[0], '%Y-%m-%d')
            feat_vec = self.X[str(day_feat_vec).split()[0]]
            prediction = self.cur_model.predict(np.reshape(feat_vec, (1, -1)))
            print(prediction[0])
            try:
                print(self.df.get_value(day_formatted1, serial_num))
                print(self.df.index)
                print(self.df[day_formatted1][serial_num])
                print(prediction)
            except:
                self.cur_prices[day] = prediction[0]
        print(self.cur_prices)
        print(len(self.cur_prices.keys()))

    """
    Function to load X which contains the feature vectors
    """
    def load_X(self, serial_num):
        self.X, params = pickle.load(open("pickled_files/training_data/sd_X_" + str(serial_num) + ".pkl", 'rb'))
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
                db_time = str(db_time)
                price = float(price)
                formatted_data[db_time] = price

        # Update the pandas dataframe
        for df_time, data in self.df.iterrows():
            for db_time in formatted_data.keys():
                if str(df_time) == str(db_time):
                    self.df.set_value(df_time, 'actual', formatted_data[db_time])

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
        if filename == None:
            self.df = pickle.load(open('pickled_files/prediction_machine/predictions.pkl', 'rb'))
            return self.df
        else:
            self.df = pickle.load(open(filename, 'rb'))
            return self.df

    """
    Function to store the database
    """
    def dump(self, filename=None):
        if filename == None:
            pickle.dump(self.df, open('pickled_files/prediction_machine/predictions.pkl', 'wb'))
        else:
            pickle.dump(self.df, open(filename, 'wb'))

    """
    Function to export the database to an excel file for further analysis
    """
    def export_excel(self, filename=None):
        start_date = str(datetime.date.today())
        start_date.replace('/', '_')
        if filename == None:
            writer = pd.ExcelWriter('analysis/predictions_' + start_date + '.xlsx')
            self.df.to_excel(writer, 'Predictions')
            writer.save()
        else:
            writer = pd.ExcelWriter(filename)
            self.df.to_excel(writer, 'Predictions')
            writer.save()

    """
    Function to store a backup of the database
    """
    def backup_db(self, filename=None):
        start_date = str(datetime.date.today())
        start_date.replace('/', '_')
        if filename == None:
            pickle.dump(self.df, open('pickled_files/prediction_machine/predictions_' + start_date + '.pkl', 'wb'))
        else:
            pickle.dump(self.df, open(filename, 'wb'))

if __name__ == '__main__':
    pm = PredictionMachine('aapl')
    pm.load()
    pm.load_model_db()
    pm.predict_models()
    #pm.update_actual()
    #pm.export_excel()
