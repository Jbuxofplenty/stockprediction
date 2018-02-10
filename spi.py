#!/usr/bin/env python3

print("Loading...")

import os
import sys
import glob
import pickle
import MySQLdb
import argparse
import datetime
import requests
import subprocess
import pandas as pd
from datetime import timedelta
from model_db import ModelDatabase
from mlp_regression import Numbers
from mlp_regression import MLPRegressor
from stock_price_data import StockPriceData
from feature_vector import FeatureVectorizor
from linear_regression import LinearRegressor

class spinterface():
    """
    Class to display an interface in the terminal and provide a user with a set of actions
    """
    def __init__(self, args):
        self.args = args
        self.categories = {1:"data", 2:"feat_vecs", 3:"models", 4:"predictions", 5:"exit"}
        self.category = -1
        self.sub_category = 0
        self.symbols = []

    """
    Helper Function to parse in the sub-category user input into the self.sub_category variable
    """
    def parse_sb_input(self, user_input):
        # parse in the user input
        try:
            self.sub_category = int(input(user_input))
        except:
            self.sub_category = 0

    """
    Helper Function to load in the symbols to grab data for
    """
    def load_symbols(self, fname='symbols'):
        # find the last pickled file and load this symbol
        list_of_files = glob.glob('pickled_files/symbols/*.pkl')
        latest_file = max(list_of_files, key=os.path.getctime)
        if fname == 'symbols':
            with open(latest_file, 'rb') as f:
                self.symbols = pickle.load(f)
        else:
            start_date = str(datetime.date.today())
            start_date.replace('/', '_')
            with open('pickled_files/symbols/' + fname + '_' + start_date + '.pkl', 'rb') as f:
                self.symbols = pickle.load(f)

    """
    Helper Function to store the symbols to a specific format
    """
    def dump_symbols(self, fname='symbols'):
        start_date = str(datetime.date.today())
        start_date.replace('/', '_')
        with open('pickled_files/symbols/' + fname + '_' + start_date + '.txt', 'w') as f:
            for symbol in self.symbols:
                f.write(symbol + '\n')
        with open('pickled_files/symbols/' + fname + '_' + start_date + '.pkl', 'wb') as f:
            pickle.dump(self.symbols, f)

    """
    Function to grab data for a symbol, create a feature vector and model,
    and store them in a database
    """
    def create_model(self):
        self.symbols = []
        X_params = {}
        X_params['table'] = input("Ticker symbol:  ")
        self.symbols.append(X_params['table'])
        model = input("Model type (mlpr, lr):  ")
        X_params['days_out_prediction'] = int(input("Number of days to predict out:  "))
        total_points = int(input("Length of X:  "))

        # non-customizable params
        X_params['start_date'] = datetime.date.today()
        X_params['time_intervals_bool'] = True
        X_params['time_intervals'] = [1]
        X_params['sector_info'] = True
        fv = FeatureVectorizor(params=X_params)

        # done because stock_price_data script loads this file to see what data to grab
        self.dump_symbols(X_params['table'])

        # grab the data for the specified stock online before creating the feature vector
        exec(open("./stock_price_data.py").read())
        print('Retrieved the data from alphavantage...')

        # cycle through the number of days at the given step size to make X and Y
        for i in range(0, total_points):
            feature_vector, output = fv.gen_feature_vector()
            fv.start_date -= timedelta(days=1)

        # store the X and Y vectors into files
        fv.dump_X()
        fv.dump_Y()
        print('Stored the appropriate feature vectors...')

        # open the new updated pickle file for training data
        fname_X = "pickled_files/training_data/sd_X_" + X_params['table'] + ".pkl"
        fname_Y = "pickled_files/training_data/sd_Y_" + X_params['table'] + ".pkl"
        data = Numbers(fname_X=fname_X, fname_Y=fname_Y)

        # make a new MLP Regressor with the optimal parameters
        if model == 'lr':
            r = LinearRegressor(train_x=data.train_x, train_y=data.train_y, test_x=data.test_x, test_y=data.test_y)
        elif model == 'mlpr':
            r = MLPRegressor(train_x=data.train_x, train_y=data.train_y, test_x=data.test_x, test_y=data.test_y)
        # train the model
        r.train()
        print('New model trained...')
        # store the model parameters into the model_db
        r.store_model_db(data, fname_X)

        # store the model with the appended serial_number
        if model == 'lr':
            fname_model = "pickled_files/models/lr_regression_" + str(r.serial_num) + "_" + X_params['table'] + ".pkl"
        elif model == 'mlpr':
            fname_model = "pickled_files/models/mlp_regression_" + str(r.serial_num) + "_" + X_params['table'] + ".pkl"
        r.dump(fname_model)

        # store the data that trained the model
        data.dump_X(serial_num=r.serial_num)
        data.dump_Y(serial_num=r.serial_num)
        print('New model stored successfully...')

    """
    Function to delete a model out of a database
    """
    def delete_model(self, serial_num):
        model_db = ModelDatabase()
        model_db.load()
        try:
            model_db.del_item(serial_num)
            os.chdir('pickled_files/models/')
            for name in glob.glob('*_' + str(serial_num) + '_*.pkl'):
                os.remove(name)
            os.chdir('..')
            os.chdir('training_data/')
            for name in glob.glob('*_' + str(serial_num) + '_*.pkl'):
                os.remove(name)
            os.chdir('..')
            os.chdir('..')
            model_db.dump()
        except:
            print(str(serial_num) + " doesn't exist.")


    """
    Function to delete all of the models out of the database
    """
    def delete_all_models(self):
        model_db = ModelDatabase()
        model_db.load()
        model_db.del_all_items()
        os.chdir('pickled_files/models/')
        for name in glob.glob('*.pkl'):
            os.remove(name)
        os.chdir('..')
        os.chdir('training_data/')
        for name in glob.glob('*.pkl'):
            os.remove(name)
        os.chdir('..')
        os.chdir('..')
        model_db.dump()

    """
    Function to display the models sub-category menu and perform the appropriate function
    """
    def models_menu(self):
        mm = """---------------------------
        Models Menu
---------------------------
    1.  Create model
    2.  Delete model
    3.  Show model parameters
    ..  Previous menu
"""

        self.parse_sb_input(mm)
        # do the required set of sequences for the option chosen

        # create a model
        if self.sub_category == 1:
            self.create_model()

        # delete a model
        elif self.sub_category == 2:
            model_db = ModelDatabase()
            model_db.load()
            serial_num = int(input("Serial Number ('0': all):  "))
            if serial_num:
                self.delete_model(serial_num)
            else:
                choice = int(input("Are you sure (y or n):  "))
                if choice == 'y':
                    self.delete_all_models()

        # print out parameters of the different models
        elif self.sub_category == 3:
            model_db = ModelDatabase()
            model_db.load()
            serial_num = int(input("Serial Number ('0': all):  "))
            if serial_num:
                try:
                    print(model_db.db[serial_num])
                except:
                    print(str(serial_num) + " doesn't exist.")
            else:
                print(model_db.db)

    """
    Helper function to create a batch script and run it given a string
    """
    def run_batch_script(self, script):
        fname = "test.bat"
        with open(fname, "w") as f:
            f.write(script)
        subp = subprocess.Popen(fname, shell=True)
        subp.communicate()
        os.remove(fname)

    """
    Function to predict the prices for today on all of the models in the db
    """
    def predict_prices_today(self, table=None):
        spd = StockPriceData()
        spd.update_symbols()
        if table is None:
            exec(open("./stock_price_data.py").read())
            print("Price data retrieved and stored into mySQL database...")
            exec(open("./prediction_machine.py").read())
            print("Predict all of the prices for the symbols in the database...")
        else:
            self.run_batch_script("python stock_price_data.py --table " + table)
            print("Price data retrieved and stored into mySQL database for table (" + table + ")...")
            self.run_batch_script("python prediction_machine.py --table " + table)
            print("Predict all of the prices for " + table + " in the database...")

    """
    Function to display the predictions sub-category menu and perform the appropriate function
    """
    def predictions_menu(self):
        pm = """---------------------------
     Predictions Menu
---------------------------
    1.  Predict today's prices
    2.  Update models for today's prices
    ..  Previous menu
"""

        self.parse_sb_input(pm)
        # do the required set of sequences for the option chosen
        if self.sub_category == 1:
            table = str(input("Table ('0': all):  "))
            if table == '0':
                self.predict_prices_today()
            else:
                self.predict_prices_today(table=table)

    """
    Function to display the feature vectors sub-category menu and perform the appropriate function
    """
    def feat_vecs_menu(self):
        fvm = """---------------------------
        Feature Vectors Menu
---------------------------
    1.  Create model
    ..  Previous menu
"""

        self.parse_sb_input(fvm)
        # do the required set of sequences for the option chosen

    """
    Function to display the data sub-category menu and perform the appropriate function
    """
    def data_menu(self):
        dm = """---------------------------
        Data Menu
---------------------------
    1.  Retrieve data
    ..  Previous menu
"""

        self.parse_sb_input(dm)
        # do the required set of sequences for the option chosen

    """
    Function to branch off to the different display menus
    """
    def sub_categories(self):
        # while the user is still in the sub-menu
        self.sub_category = -1
        while self.sub_category:
            if self.category == 1:
                self.data_menu()
            elif self.category == 2:
                self.feat_vecs_menu()
            elif self.category == 3:
                self.models_menu()
            elif self.category == 4:
                self.predictions_menu()
            else:
                self.sub_category = 0
                self.main_menu

    """
    Function to display the main menu and all of the sub-categories
    Returns the category selected after progressing through the sub-menu structure
    """
    def main_menu(self):
        mm = """---------------------------
Stock Prediction Interface
---------------------------
    1.  Data
    2.  Feature Vectors
    3.  Models
    4.  Predictions
    ..  Exit
"""
        try:
            self.category = int(input(mm))
        except:
            self.category = 0
        if not self.category:
            exit()
        self.sub_categories()
        return self.category

def main():
    parser = argparse.ArgumentParser(description='Stock Price Interface options')
    parser.add_argument('--version', type=float, default=1.0,
                        help="Version number of the stock prediction system")
    parser.add_argument('--init', type=int, default=1,
                        help="Initial category to enter")
    args = parser.parse_args()

    # init
    spi = spinterface(args)
    choice = -1

    # print the main menu and cycle through user input
    while(choice):
        choice = spi.main_menu()

if __name__ == "__main__":
    main()
