#!/usr/bin/env python3

print("Loading...")

import os
import sys
import glob
import pickle
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

"""
Helper Function to store the symbols to a specific format
"""
def dump_symbols(symbols, fname='symbols'):
    start_date = str(datetime.date.today())
    start_date.replace('/', '_')
    with open('pickled_files/symbols/' + fname + '_' + start_date + '.txt', 'w') as f:
        if type(symbols) != type([]):
            symbols = [symbols]
        for symbol in symbols:
            f.write(symbol + '\n')
    with open('pickled_files/symbols/' + fname + '_' + start_date + '.pkl', 'wb') as f:
        pickle.dump(symbols, f)

"""
Helper function that loads the stocks from the last edited file, pass a table to only load a single stock
"""
def load_symbols(fname='symbols'):
    with open('pickled_files/symbols/' + fname + '.txt', 'r') as f:
        return [l.strip().lower() for l in f]

"""
Function to grab data for a symbol, create a feature vector and model,
and store them in a database
"""
def create_models():
    tables = load_symbols()
    print(tables)
    X_params = {}
    for table in tables:
        print(table, ' table generation in progress...')
        X_params['table'] = table
        model = 'lr'
        X_params['days_out_prediction'] = 7
        total_points = 500
        X_params['start_date'] = datetime.date.today()
        X_params['sector_info'] = False
        X_params['type_options'] = {'STOCH':True, 'MACD':True, 'RSI':True, 'EMA':True, 'SMA':True, 'TIME_SERIES_DAILY_ADJUSTED':True}

        fv = FeatureVectorizor(params=X_params)

        # done because stock_price_data script loads this file to see what data to grab
        dump_symbols(table, table)

        # grab the data for the specified stock online before creating the feature vector
        exec(open("./stock_price_data.py").read())
        print('Retrieved the data from alphavantage...')

        # load the data into the class
        fv.load_sp_data()

        # cycle through the number of days at the given step size to make X and Y
        potential_points = len(fv.sp_data['TIME_SERIES_DAILY_ADJUSTED'].keys()) - 100
        if potential_points <= 0:
            print('No stock data available...')
            # delete the generated symbol files
            start_date = str(datetime.date.today())
            start_date.replace('/', '_')
            list_of_files = glob.glob('pickled_files/symbols/*'+ start_date + '*')
            for file in list_of_files:
                os.remove(file)
            continue
        if total_points > potential_points:
            total_points = potential_points
        for i in range(0, total_points):
            feature_vector, output = fv.gen_feature_vector()
            fv.start_date -= timedelta(days=1)

        # store the X and Y vectors into files
        fv.dump_X()
        fv.dump_Y()
        print('Stored the appropriate feature vectors...')

        # open the new updated pickle file for training data
        fname_X = "pickled_files/training_data/" + X_params['table'][0] + "/sd_X_" + X_params['table'] + ".pkl"
        fname_Y = "pickled_files/training_data/" + X_params['table'][0] + "/sd_Y_" + X_params['table'] + ".pkl"
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

        # delete the generated symbol files
        start_date = str(datetime.date.today())
        start_date.replace('/', '_')
        list_of_files = glob.glob('pickled_files/symbols/*'+ start_date + '*')
        for file in list_of_files:
            os.remove(file)

create_models()
