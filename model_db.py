import pickle
import hashlib
import datetime
import numpy as np
import pandas as pd

class ModelDatabase:
    """
    Class to load and store the machine learning model database for stock prices
    """
    def __init__(self):
        self.db = {}
        self.cur_model_params = {}
        self.columns = ['type', 'num_days', 'model_params', 'num_train', 'num_test', 'num_features', 'X_params', 'model_hash', 'X_hash', 'news_params']
        for column in self.columns:
            self.cur_model_params[column] = None

    """
    Function to store data into the data base
    """
    def store_data(self, serial_number):
        if not serial_number in self.db:
            self.db[serial_number] = {}
        for col, datum in zip(self.cur_model_params.keys(), self.cur_model_params.values()):
            if datum is not None:
                self.db[serial_number][col] = datum

    """
    Function to store data into the current model
    """
    def store_cur_data(self, data, columns='all'):
        if columns == 'all':
            columns = self.columns
        for col, datum in zip(columns, data):
            self.cur_model_params[col] = datum

    """
    Function to see if the cur_model_params variable is completely filled
    """
    def is_cur_data_filled(self):
        for datum in self.cur_model_params.values():
            if datum is None:
                return False
        return True

    """
    Function to find the next serial number for a newly added model to the database
    """
    def find_serial_number(self):
        new_index = 1
        for key in self.db.keys():
            is_same = True
            for col, datum in zip(self.cur_model_params.keys(), self.cur_model_params.values()):
                if self.db[key][col] != datum and col != 'model_hash':
                    is_same = False
            if is_same:
                return new_index
            else:
                new_index += 1
        return new_index

    """
    Function to find the hash of a given file
    """
    def find_hash(self, filename):
        return hashlib.md5(open(filename, 'rb').read()).hexdigest()

    """
    Function to load a previously stored database
    """
    def load(self, filename=None):
        if filename == None:
            self.db = pickle.load(open("pickled_files/model_dbs/model_db.pkl", 'rb'))
            return self.db
        else:
            self.db = pickle.load(open(filename, 'rb'))
            return self.db

    """
    Function to store the database
    """
    def dump(self, filename=None):
        if filename == None:
            pickle.dump(self.db, open("pickled_files/model_dbs/model_db.pkl", 'wb'))
        else:
            pickle.dump(self.db, open(filename, 'wb'))

    """
    Function to store a backup of the database
    """
    def backup_db(self, filename=None):
        start_date = str(datetime.date.today())
        start_date.replace('/', '_')
        if filename == None:
            pickle.dump(self.db, open("pickled_files/model_dbs/model_db_" + start_date + ".pkl", 'wb'))
        else:
            pickle.dump(self.db, open(filename, 'wb'))

if __name__ == '__main__':
    model_db = ModelDatabase()
    model_db.load()
    model_db.update_models()
    model_db.dump()
