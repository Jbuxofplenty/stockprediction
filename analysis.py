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

class AnalysisMachine:
    """
    Class to analyze the pandas dataframes created by prediction_machine.py and create a database of associated a
    """
    def __init__(self, stock):
        self.df = None
        self.table = stock
        self.model_db = None

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
    Helper function to find the last saved pickle file of the prediction machine
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
    parser = argparse.ArgumentParser(description='Analysis options')
    parser.add_argument('--table', type=str, default=None,
                        help="Pass in a specific table to analyze its predictions")
    args = parser.parse_args()

    if args.table is None:
        # find the last pickled file and load this set of symbols
        list_of_files = glob.glob('pickled_files/symbols/*.pkl')
    else:
        # find the last pickled file and load this set of symbols
        list_of_files = glob.glob('pickled_files/symbols/' + args.table + '*.pkl')
    fname = max(list_of_files, key=os.path.getctime)
    with open(fname, 'rb') as f:
        tables = pickle.load(f)
        for table in tables:
            am = AnalysisMachine(table)
