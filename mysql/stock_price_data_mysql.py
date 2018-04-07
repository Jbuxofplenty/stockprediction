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

class StockPriceData():
    """
    Class that grabs stock price data through an Alphavantage API and has the ability to store data in a dict and on a MySQL DB
    """
    def __init__(self, base_url='https://www.alphavantage.co/query?', apikey='623V97UCQMEHDMXX'):
        self.stocks = {}
        self.header = {'User-Agent': 'Chrome/62.0.3202.62'}
        self.base_url = base_url
        self.symbols = []
        self.functions = ['TIME_SERIES_INTRADAY', 'TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED', 'TIME_SERIES_WEEKLY', 'TIME_SERIES_WEEKLY_ADJUSTED', 'TIME_SERIES_MONTHLY', 'TIME_SERIES_MONTHLY_ADJUSTED']
        self.stock_data_categories = []
        self.time_period = ['10', '50', '100', '200']
        self.series_type = ['close', 'open', 'high', 'low', 'volume']
        self.intervals = ['1min', '5min', '15min', '30min', '60min', 'daily']
        self.columns = ['price', 'SMA', 'EMA', 'MACD', 'MACD_HIST', 'MACD_SIGNAL', 'STOCH_K', 'STOCH_D', 'RSI']
        self.sector_time_periods = ['Real-Time', '1 Day', '5 Day', '1 Month', '3 Month', 'YTD', '1 Year', '3 Year', '5 Year', '10 Year']
        self.sectors = ['Information Technology', 'Health Care', 'Materials', 'Financials', 'Consumer Discretionary', 'Industrials', 'Consumer Staples', 'Utilities', 'Real Estate', 'Energy', 'Telecommunication Services']
        self.apikey = apikey

        # Store data into a dict as well
        self.sp_data = {}
        for symbol in self.symbols:
            self.sp_data[symbol] = {}

    """
    Function to retrieve and parse the data from the alphavantage api
    """
    def retrieve_stock_data(self, url, function):
        response = requests.get(url, headers=self.header)
        content = response.content.decode('utf-8')
        data = {}

        if function in self.functions:
            if function == 'TIME_SERIES_INTRADAY':
                matches = re.findall('(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).+?[\n]+.+": "(\d+.\d+)".+?[\n]+.+": "(\d+.\d+)".+?[\n]+.+": "(\d+.\d+)".+?[\n]+.+": "(\d+.\d+)"', content)
            else:
                matches = re.findall('(\d{4}-\d{2}-\d{2}).+?[\n]+.+": "(\d+.\d+)".+?[\n]+.+": "(\d+.\d+)".+?[\n]+.+": "(\d+.\d+)".+?[\n]+.+": "(\d+.\d+)"', content)
            for match in matches:
                for i, datum in enumerate(match):
                    if i:
                        value = Decimal(float(datum))
                        value = round(value, 5)
                        data[time].append(value)
                    else:
                        time = datum
                        data[time] = []

        if function == 'MACD':
            matches = re.findall('(\d{4}-\d{2}-\d{2}).+?[\n]*.*?": "(-?\d*\.{0,1}\d+)".+?[\n]*.*?": "(-?\d*\.{0,1}\d+)".+?[\n]*.*?": "(-?\d*\.{0,1}\d+)"', content)
            for match in matches:
                for i, datum in enumerate(match):
                    if i:
                        value = round(float(datum), 5)
                        data[time].append(value)
                    else:
                        time = datum
                        data[time] = []

        if function == 'STOCH':
            matches = re.findall('(\d{4}-\d{2}-\d{2}).+?[\n]*.*?": "(-?\d*\.{0,1}\d+)".+?[\n]*.*?": "(-?\d*\.{0,1}\d+)"', content)
            for match in matches:
                for i, datum in enumerate(match):
                    if i:
                        value = round(float(datum), 5)
                        data[time].append(value)
                    else:
                        time = datum
                        data[time] = []

        if function == 'SMA' or function == 'EMA' or function == 'RSI':
            matches = re.findall('(\d{4}-\d{2}-\d{2}).*?[\n]*.+": "(\d+.\d+)', content)
            for match in matches:
                for i, datum in enumerate(match):
                    if i:
                        value = round(float(datum), 5)
                        data[time].append(value)
                    else:
                        time = datum
                        data[time] = []

        if function == 'SECTOR':
            matches = re.findall('Performance": {[\n\s"]*(.*?)":\s"(.*?)",\n[\s]*"(.*?)":\s"(.*?)",\n[\s]*"(.*?)":\s"(.*?)",\n[\s]*"(.*?)":\s"(.*?)",\n[\s]*"(.*?)":\s"(.*?)",\n[\s]*"(.*?)":\s"(.*?)",\n[\s]*"(.*?)":\s"(.*?)",\n[\s]*"(.*?)":\s"(.*?)",\n[\s]*"(.*?)":\s"(.*?)",\n[\s]*"(.*?)":\s"(.*?)",\n[\s]*"(.*?)":\s"(.*?)"', content)
            now = str(datetime.date.today())
            for i, match in enumerate(matches):
                for j, datum in enumerate(match):
                    if j:
                        if j % 2:
                            data[now + ' 00:00:' + str(i).zfill(2)][index+1] = datum
                        else:
                            index = self.sectors.index(datum)
                    else:
                        data[now + ' 00:00:' + str(i).zfill(2)] = [None] * (len(self.sectors) + 1)
                        data[now + ' 00:00:' + str(i).zfill(2)][0] = self.sector_time_periods[i]
                        index = self.sectors.index(datum)
        return data

    """
    Function to store data into the stock price data dictionary
    """
    def store_dict(self, data, table):
        if not table in self.sp_data.keys():
            self.sp_data[table] = {}
        for key in data.keys():
            if not key in self.sp_data[table].keys():
                self.sp_data[table][key] = data[key]

    """
    Function to load data into the mysql database, given queries
    """
    def update_db(self, queries):
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
    Function to check to see if the tables are present in the db
    """
    def are_tables(self):
        query = "Show Tables;"
        data = list(self.db_retrieve_data(query))
        for i, datum in enumerate(data):
            data[i] = re.sub('[(),\']', '', str(datum))
        table_exists = []
        for symbol in self.symbols:
            if symbol in data:
                table_exists.append(True)
            else:
                table_exists.append(False)
        return table_exists

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
    Function to delete any of the rows in the database given a table that does not have each column filled
    """
    def clean_db(self, table):
        url = 'DELETE FROM ' + table + ' WHERE (price IS NULL OR SMA IS NULL OR EMA IS NULL OR MACD IS NULL OR MACD_HIST IS NULL OR MACD_SIGNAL IS NULL OR STOCH_D IS NULL OR STOCH_K IS NULL OR RSI IS NULL);'
        self.update_db([url])

    """
    Function to create a query that will insert records into the mysql database
    """
    def build_query(self, data, table='aapl', column='price', columns=[]):
        queries = []
        for key in data.keys():
            if column == 'MACD' or column == 'STOCH' or column == 'all':
                column_query = '(timestamp, '
                value_query = ' VALUES ("' + key + '", '
                on_duplicate_query = ' ON DUPLICATE KEY UPDATE '
                for i, col in enumerate(columns):
                    if i == len(columns) - 1:
                        column_query += col + ')'
                        value_query += str(data[key][i]) + ')'
                        on_duplicate_query += columns[i] + '=VALUES(' + columns[i] + ');'
                    else:
                        column_query += col + ', '
                        value_query += str(data[key][i]) + ','
                        on_duplicate_query += columns[i] + '=VALUES(' + columns[i] + '),'
                queries.append('INSERT INTO ' + table + ' ' + column_query + value_query + on_duplicate_query)
            else:
                column_query = '(timestamp, ' + column + ')'
                if column == 'price':
                    value = (data[key][0] + data[key][3]) / 2
                if column == 'SMA' or column == 'EMA' or column == 'RSI':
                    value = data[key][0]
                value_query = '("' + key + '", ' + str(value) + ')'
                queries.append('INSERT INTO ' + table + ' ' + column_query + ' VALUES ' + value_query + ' ON DUPLICATE KEY UPDATE ' + column + '=VALUES(' + column + ');')
        return queries

    """
    Function to create a query that will insert sector records into the mysql database
    """
    def build_sector_query(self, data, table='SECTOR', column='SECTOR', columns=['information_technology', 'health_care', 'materials', 'financials', 'consumer_discretionary', 'industrials', 'consumer_staples', 'utilities', 'real_estate', 'energy', 'telecommunication_services']):
        queries = []
        for key in data.keys():
            column_query = '(timestamp, '
            value_query = ' VALUES ("' + key + '", '
            on_duplicate_query = ' ON DUPLICATE KEY UPDATE '
            for i, col in enumerate(columns):
                if i == len(columns) - 1:
                    column_query += col + ')'
                    value = float(data[key][i].strip('%'))/100
                    value = round(float(value), 5)
                    value1 = float(data[key][i+1].strip('%'))/100
                    value1 = round(float(value1), 5)
                    value_query += str(value) + ',' + str(value1) + ')'
                    on_duplicate_query += columns[i] + '=VALUES(' + columns[i] + ');'
                else:
                    if i:
                        column_query += col + ', '
                        value = float(data[key][i].strip('%'))/100
                        value = round(float(value), 5)
                        value_query += str(value) + ','
                        on_duplicate_query += columns[i] + '=VALUES(' + columns[i] + '),'
                    else:
                        column_query += 'time_period, ' + col + ', '
                        value_query += '"' + data[key][i] + '",'
                        on_duplicate_query += 'time_period=VALUES(time_period),'

            queries.append('INSERT INTO ' + table + ' ' + column_query + value_query + on_duplicate_query)
        return queries

    """
    Helper function that creates an api request to be sent to alpha vantage (ma)
    """
    def build_MA_url(self, function=None, interval=None, output_size='full', symbol=None, time_period='50', series_type='close'):
        url = self.base_url + 'function=' + function + '&symbol=' + symbol + '&interval=' + interval + '&time_period=' + time_period + '&series_type=' + series_type + '&apikey=' + self.apikey
        return url

    """
    Helper function that creates an api request to be sent to alpha vantage (macd)
    """
    def build_MACD_url(self, function=None, interval=None, symbol=None, series_type='close'):
        url = self.base_url + 'function=' + function + '&symbol=' + symbol + '&interval=' + interval + '&series_type=' + series_type + '&apikey=' + self.apikey
        return url

    """
    Helper function that creates an api request to be sent to alpha vantage (stoch)
    """
    def build_STOCH_url(self, function=None, interval=None, symbol=None):
        url = self.base_url + 'function=' + function + '&symbol=' + symbol + '&interval=' + interval + '&apikey=' + self.apikey
        return url

    """
    Helper function that creates an api request to be sent to alpha vantage (time_series)
    """
    def build_time_series_url(self, function=None, interval=None, output_size='full', symbol=None):
        if function == 'TIME_SERIES_INTRADAY':
            url = self.base_url + 'function=' + function + '&symbol=' + symbol + '&interval=' + interval + '&outputsize=' + output_size + '&apikey=' + self.apikey
        elif function == 'TIME_SERIES_DAILY' or function == 'TIME_SERIES_DAILY_ADJUSTED':
            url = self.base_url + 'function=' + function + '&symbol=' + symbol + '&outputsize=' + output_size + '&apikey=' + self.apikey
        else:
            url = self.base_url + 'function=' + function + '&symbol=' + symbol + '&apikey=' + self.apikey
        return url

    """
    Helper function that creates a create table query given a table for the mySQL db
    """
    def build_create_table_query(self, table):
        query = 'CREATE TABLE ' + table + '(timestamp DATETIME NOT NULL PRIMARY KEY, price DECIMAL(20,5), SMA DECIMAL (20,5), EMA DECIMAL (20,5), MACD DECIMAL (20,5), MACD_HIST DECIMAL (20,5), MACD_SIGNAL DECIMAL (20,5), STOCH_K DECIMAL (20,5), STOCH_D DECIMAL (20,5), RSI DECIMAL (20,5));'
        return query

    """
    Helper function that creates a drop table query given a table for the mySQL db
    """
    def build_drop_table_query(self, table):
        query = 'DROP TABLE ' + table + ';'
        return query

    """
    Function that parses in the information stored in the stock csv files
    """
    def load_symbols_csv(self, fnames=['init_files/NASDAQ.csv', 'init_files/NYSE.csv', 'init_files/AMEX.csv']):
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
                                self.stocks[symbol][self.stock_data_categories[j]] = datum
                            else:
                                self.symbols.append(datum)
                                self.stocks[datum] = {}
                                symbol = datum
                    else:
                        for category in line_split:
                            if not category == '\n' and not category in self.stock_data_categories:
                                self.stock_data_categories.append(category)

    """
    Function that calculates the averages of each of the columns given a given range
    """
    def update_daily_avgs(self, table='aapl', time_interval=7):
        start_timestamp = datetime.date.today()
        data = {}
        for i in range(time_interval):
            end_timestamp = start_timestamp - timedelta(days=1)
            query = 'SELECT * FROM ' + table + ' WHERE timestamp < \'' + str(start_timestamp) + '\' AND timestamp > \'' + str(end_timestamp) + '\';'
            datum = self.db_retrieve_data(query=query)
            if datum != None:
                data[str(end_timestamp)] = datum
            start_timestamp -= timedelta(days=1)

        # Average the values in each of the columns
        for key in data.keys():
            temp = [0] * len(data[key][0])
            num_rows = len(data[key])
            for row in data[key]:
                for i, item in enumerate(row):
                    temp[i] += item
            for i, item in enumerate(temp):
                temp[i] = round(item / num_rows, 5)
            data[key] = temp
        queries = self.build_query(data, table='aapl', column='all', columns=self.columns)
        self.update_db(queries=queries)

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

def main():
    parser = argparse.ArgumentParser(description='Stock Price Data options')
    parser.add_argument('--table', type=str, default=None,
                        help="Pass in a specific table to grab data for, not specifying grabs data for all the tables.")
    args = parser.parse_args()
    sp_data = StockPriceData()
    sp_data.update_symbols()
    sp_data.load_symbols('all_symbols')
    all_symbols = sp_data.symbols
    if args.table is None:
        sp_data.load_symbols()
    else:
        sp_data.load_symbols(args.table)

    # Init
    # sp_data.load_functions()
    # sp_data.load_symbols_csv()
    # sp_data.symbols = ['aapl', 'cmg', 'fix']
    sp_data.dump_symbols()
    # quit()

    # Cycle through each of the symbols and create a table for them
    table_exists = sp_data.are_tables()
    for i, symbol in enumerate(sp_data.symbols):
        if not table_exists[i]:
            query = sp_data.build_create_table_query(symbol)
            sp_data.update_db(queries=[query])

    sp_data.load_sp_data()

    # Cycle through each of the symbols and grab data for them
    for symbol in sp_data.symbols:
        # Check to ensure the tables isn't already present in the db, change output_size accordingly
        if symbol in all_symbols:
            output_size = 'compact'
        else:
            output_size = 'full'
        # Price
        functions = ['TIME_SERIES_DAILY_ADJUSTED']
        for function in functions:
            url = sp_data.build_time_series_url(function=function, interval='daily', symbol=symbol, output_size=output_size)
            data = sp_data.retrieve_stock_data(url=url, function=function)
            sp_data.store_dict(data=data, table=symbol)
            queries = sp_data.build_query(data=data, table=symbol, column='price')
            sp_data.update_db(queries=queries)

            # SMA
            url = sp_data.build_MA_url(function='SMA', interval='daily', symbol=symbol, output_size=output_size)
            data = sp_data.retrieve_stock_data(url=url, function='SMA')
            sp_data.store_dict(data=data, table=symbol)
            queries = sp_data.build_query(data=data, table=symbol, column='SMA')
            sp_data.update_db(queries=queries)

            # EMA
            url = sp_data.build_MA_url(function='EMA', interval='daily', symbol=symbol, output_size=output_size)
            data = sp_data.retrieve_stock_data(url=url, function='EMA')
            sp_data.store_dict(data=data, table=symbol)
            queries = sp_data.build_query(data=data, table=symbol, column='EMA')
            sp_data.update_db(queries=queries)

            # MACD
            url = sp_data.build_MACD_url(function='MACD', interval='daily', symbol=symbol)
            data = sp_data.retrieve_stock_data(url=url, function='MACD')
            sp_data.store_dict(data=data, table=symbol)
            queries = sp_data.build_query(data=data, table=symbol, column='MACD', columns=['MACD', 'MACD_HIST', 'MACD_SIGNAL'])
            sp_data.update_db(queries=queries)

            # STOCH
            url = sp_data.build_STOCH_url(function='STOCH', interval='daily', symbol=symbol)
            data = sp_data.retrieve_stock_data(url=url, function='STOCH')
            sp_data.store_dict(data=data, table=symbol)
            queries = sp_data.build_query(data=data, table=symbol, column='STOCH', columns=['STOCH_K', 'STOCH_D'])
            sp_data.update_db(queries=queries)

            # RSI
            url = sp_data.build_MA_url(function='RSI', interval='daily', symbol=symbol, output_size=output_size)
            data = sp_data.retrieve_stock_data(url=url, function='RSI')
            sp_data.store_dict(data=data, table=symbol)
            queries = sp_data.build_query(data=data, table=symbol, column='RSI')
            sp_data.update_db(queries=queries)

            # Sector Performance (US)
            url = sp_data.base_url + 'function=SECTOR&apikey=' + sp_data.apikey
            data = sp_data.retrieve_stock_data(url=url, function='SECTOR')
            sp_data.store_dict(data=data, table=symbol)
            queries = sp_data.build_sector_query(data=data)
            sp_data.update_db(queries=queries)

    sp_data.dump_sp_data()
    # print(sp_data.sp_data)

if __name__ == '__main__':
    main()
