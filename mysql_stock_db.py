#!/usr/bin/env python3

import re
import os
import csv
import glob
import time
import pickle
import urllib
import MySQLdb
import requests
import datetime
import argparse
from decimal import Decimal
from datetime import timedelta

class StockDB():
    """
    Class that manages the MySQL database "stock" table where all of the information on stocks is stored
    """
    def __init__(self):
        self.stocks = {}
        self.symbols = []
        self.stock_data_categories = {}
        self.sectors = ['Information Technology', 'Health Care', 'Materials', 'Financials', 'Consumer Discretionary', 'Industrials', 'Consumer Staples', 'Utilities', 'Real Estate', 'Energy', 'Telecommunication Services']

    """
    Function to retrieve data from the mysql database and format it to be returned as a list
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
            if query == "Show Tables;":
                return data
            if len(data):
               formatted_data = self.format_data(data=data)
            else:
               formatted_data = None
            # Commit your changes in the database
            db.commit()
        except (MySQLdb.Error, MySQLdb.Warning) as e:
            print(e)
            # Rollback in case there is any error
            db.rollback()
            formatted_data = None
        db.close()

        return formatted_data

    """
    Function to load data into the mysql database, given queries
    """
    def db_update_data(self, queries):
        db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                             user="josiah",         # your username
                             passwd="adversereaction",  # your password
                             db="stock_data")        # name of the data base
        cur = db.cursor()
        for query in queries:
            try:
               # Execute the SQL command
               cur.execute(query)
               # Commit your changes in the database
               db.commit()
            except (MySQLdb.Error, MySQLdb.Warning) as e:
                print(e)
                # Rollback in case there is any error
                db.rollback()
        db.close()

    """
    Function to format the retrieved data from the mysql database
    """
    def format_data(self, data=None):
        formatted_data = []
        for row in data:
            formatted_row = []
            for i, datum in enumerate(row):
                if i:
                    if datum == None:
                        formatted_row.append(0)
                    else:
                        formatted_row.append(float(datum))
            formatted_data.append(formatted_row)
        return formatted_data

    """
    Function to create a query that will insert records into the mysql database
    """
    def build_query(self, data, table='stocks', columns=[]):
        queries = []
        for key in data.keys():
            column_query = '(symbol, '
            if table == 'stocks':
                value_query = ' VALUES ("' + key + '", '
            else:
                value_query = ' VALUES ("' + key + '", '
            on_duplicate_query = ' ON DUPLICATE KEY UPDATE '
            for i, col in enumerate(columns):
                if col == 'symbol':
                    pass
                elif i == len(columns) - 1:
                    column_query += col + ')'
                    value_query += '\"' + str(data[key][col]) + '\")'
                    on_duplicate_query += columns[i] + '=VALUES(' + columns[i] + ');'
                else:
                    column_query += col + ', '
                    value_query += '\"' + str(data[key][col]) + '\",'
                    on_duplicate_query += columns[i] + '=VALUES(' + columns[i] + '),'
            queries.append('INSERT INTO ' + table + ' ' + column_query + value_query + on_duplicate_query)
        return queries

    """
    Function that parses in the information stored in the stock csv files
    """
    def load_symbols_csv(self, fnames=['init_files/NASDAQ.csv', 'init_files/NYSE.csv', 'init_files/AMEX.csv'], columns=['symbol', 'name', 'lastsale', 'marketcap', 'IPOyear', 'sector', 'industry', 'summaryquoteurl']):
        for fname in fnames:
            with open(fname, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line_split = re.split('","|,"\n|"|,\n', line)
                    line_split = list(filter(None, line_split))
                    if '.' in line_split[0] or '^' in line_split[0]:
                        continue
                    if i:
                        for j, datum in enumerate(line_split):
                            if j:
                                if j in self.stock_data_categories.keys():
                                    self.stocks[symbol][self.stock_data_categories[j]] = datum
                            else:
                                self.symbols.append(datum)
                                self.stocks[datum] = {}
                                symbol = datum
                    else:
                        for i, category in enumerate(line_split):
                            if category in columns:
                                if not category == '\n' and not category in self.stock_data_categories.values():
                                    self.stock_data_categories[i] = category

    """
    Helper function that dumps the stocks stored in self.symbols to a pickled file and text file
    """
    def dump_symbols(self, fname='symbols'):
        with open('pickled_files/symbols/' + fname + '.txt', 'w') as f:
            for symbol in self.symbols:
                f.write(symbol + '\n')
        with open('pickled_files/symbols/' + fname + '.pkl', 'wb') as f:
            pickle.dump(self.symbols, f)

    """
    Helper function that dumps the stock data stored in self.sp_data to a pickled file
    """
    def dump_sp_data(self, fname='sp_data'):
        with open('pickled_files/sp_data/' + fname + '.pkl', 'wb') as f:
            pickle.dump(self.sp_data, f)

    """
    Helper function that loads the stock data stored in self.sp_data to a dict
    """
    def load_sp_data(self, fname='sp_data'):
        with open('pickled_files/sp_data/' + fname + '.pkl', 'rb') as f:
            self.sp_data = pickle.load(f)

    """
    Helper function that loads the stocks from the last edited file, pass a table to only load a single stock
    """
    def load_symbols(self, fname='symbols', table=None):
        if table is None:
            # find the last pickled file and load this symbol
            list_of_files = glob.glob('pickled_files/symbols/*.pkl')
            latest_file = max(list_of_files, key=os.path.getctime)
            if fname == 'symbols':
                with open(latest_file, 'rb') as f:
                    self.symbols = pickle.load(f)
            else:
                with open('pickled_files/symbols/' + fname + '.pkl', 'rb') as f:
                    self.symbols = pickle.load(f)
        else:
            # find the last pickled file and load this symbol
            list_of_files = glob.glob('pickled_files/symbols/*' + table + '*.pkl')
            latest_file = max(list_of_files, key=os.path.getctime)
            if fname == 'symbols':
                with open(latest_file, 'rb') as f:
                    self.symbols = pickle.load(f)
            else:
                with open('pickled_files/symbols/' + fname + '.pkl', 'rb') as f:
                    self.symbols = pickle.load(f)

    """
    Helper function that loads a list of queries to be used for the mySQL db
    """
    def load_functions(self, fname='functions'):
        with open('init_files/' + fname + '.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.functions.append(line.split('\n')[0])

    """
    Helper function to update the symbols with all of the values in the mySQL
    """
    def update_symbols(self, fname='all_symbols'):
        query = "Show Tables;"
        data = list(self.db_retrieve_data(query))
        symbols = []
        for i, datum in enumerate(data):
            data[i] = re.sub('[(),\']', '', str(datum))
            if data[i] != 'sector' and data[i] != 'stocks':
                symbols.append(data[i])
        with open('pickled_files/symbols/' + fname + '.txt', 'w') as f:
            for symbol in symbols:
                data = [symbol]
                f.write(symbol + '\n')
                with open('pickled_files/symbols/' + symbol + '.txt', 'w') as f1:
                    f1.write(symbol)
                with open('pickled_files/symbols/' + symbol + '.pkl', 'wb') as f2:
                    pickle.dump(data, f2)
        with open('pickled_files/symbols/' + fname + '.pkl', 'wb') as f:
            pickle.dump(symbols, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Database options')
    parser.add_argument('--table', type=str, default=None,
                        help="Pass in a specific table to grab data for, not specifying grabs data for all the tables.")

    args = parser.parse_args()
    stock_db = StockDB()
    stock_db.load_symbols_csv()
    queries = stock_db.build_query(stock_db.stocks, columns=['symbol', 'name', 'lastsale', 'marketcap', 'IPOyear', 'sector', 'industry', 'summaryquoteurl'])
    stock_db.db_update_data(queries)
