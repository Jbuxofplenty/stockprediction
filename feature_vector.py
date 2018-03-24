import pickle
import urllib
import MySQLdb
import argparse
import datetime
import requests
from datetime import timedelta

class FeatureVectorizor():
    """
    Class that provides a framework for creating arrays of feature vectors that can be used to train machine learning models
    """
    def __init__(self, params):
        # Calculated variables
        self.feature_vector = []
        self.data = []
        self.output = -1

        # All of the combined feature vectors/outputs generated from the object
        self.X = {}
        self.Y = {}

        # Variable to store all of the parameters for the combined feature vectors
        self.params = params

        # Custom variables set by the designer and used throughout the class
        self.days_out_prediction = self.params['days_out_prediction']
        self.start_date = self.params['start_date']
        self.time_intervals_bool = self.params['time_intervals_bool']
        if self.time_intervals_bool:
            self.time_intervals = self.params['time_intervals']
        else:
            self.time_intervals = [1]
        self.sectors = ['information_technology', 'health_care', 'materials', 'financials', 'consumer_discretionary', 'industrials', 'consumer_staples', 'utilities', 'real_estate', 'energy', 'telecommunication_services']
        self.table = self.params['table']

    """
    Function that generates a single feature vector and appends it to X (the array of feature vecs for a model to be trained on)
    """
    def gen_feature_vector(self, start_date=None):
        # Reinitialize all of the variables
        self.feature_vector = []
        self.data = []
        self.output = -1
        if start_date is not None:
            self.start_date = start_date

        # Cycle through each of the time intervals to generate a list of parameters to be added to the feature vector
        for time_interval in self.time_intervals:
            timestamps = self.create_timestamp_list(start_date=self.start_date, time_interval=time_interval)
            data = []
            for timestamp in timestamps:
                query = self.create_query(table=self.table, timestamp=timestamp)
                retrieved_data = None
                while retrieved_data == None:
                    retrieved_data = self.db_retrieve_data(query=query)
                    timestamp -= timedelta(days=1)
                    query = self.create_query(table=self.table, timestamp=timestamp)
                data.append(retrieved_data)
            if time_interval == 1:
                self.feature_vector.append(data[0])
                self.feature_vector.append(self.find_derivative(data=data, time_interval=time_interval))
                self.format_feature_vector()
        if self.params['sector_info']:
            self.append_sector_info()
        self.find_output(table=self.table, time_interval=self.days_out_prediction)
        self.X[str(self.start_date)] = self.feature_vector
        self.Y[str(self.start_date)] = self.output
        return self.feature_vector, self.output

    """
    Function that retrieves data out of the mySQL database and returns a formatted list
    """
    def db_retrieve_data(self, query, table=None):
        db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                             user="josiah",         # your username
                             passwd="adversereaction",  # your password
                             db="stock_data")        # name of the data base
        cur = db.cursor()
        try:
            # Execute the SQL command
            cur.execute(query)
            data = cur.fetchall()
            if len(data):
                if table == 'SECTOR':
                    formatted_data = self.format_sector_data(data=data)
                else:
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
    Function that formats the data from the mySQL db to a list
    """
    def format_data(self, data=None):
        formatted_data = []
        for i, datum in enumerate(data[0]):
            if i:
                if datum == None:
                    formatted_data.append(0)
                else:
                    formatted_data.append(float(datum))
        return formatted_data

    """
    Function that formats the sector data from the mySQL db to a list
    """
    def format_sector_data(self, data=None):
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
    Function that creates a query which will grab data for a single day from the MySQL db given a table and a day
    """
    def create_query(self, table, timestamp=None, whole_day=False):
        if timestamp == None:
            query = 'SELECT * FROM ' + table + ';'
        else:
            if whole_day:
                query = 'SELECT * FROM ' + table + ' WHERE timestamp < \'' + str(timestamp+timedelta(days=1)) + '\' AND timestamp > \'' + str(timestamp) + '\';'
            else:
                query = 'SELECT * FROM ' + table + ' WHERE timestamp = \'' + str(timestamp) + '\';'
        return query

    """
    Helper function that creates a list of timestamps for the running average features
    """
    def create_timestamp_list(self, start_date=None, time_interval=1):
        if start_date is None:
            start_date = self.start_date
        timestamps = [start_date]
        end_date = start_date - timedelta(days=time_interval)
        timestamps.append(end_date)
        return timestamps

    """
    Helper function that finds the derivative of a set of data given a time_interval
    """
    def find_derivative(self, data=None, time_interval=1):
        feature_vector = []
        for end_datum, start_datum in zip(data[0], data[1]):
            feature_vector.append((end_datum - start_datum) / time_interval)
        self.feature_vector.append(feature_vector)
        return feature_vector

    """
    Helper function that finds the correct output to append to self.Y given a time interval, date, and table
    """
    def find_output(self, start_date=None, time_interval=7, table=None):
        if start_date is None:
            start_date = self.start_date
        timestamp = start_date + timedelta(days=time_interval)
        query = 'SELECT timestamp, price FROM ' + table + ' WHERE timestamp = \'' + str(timestamp) + '\';'
        retrieved_data = None
        while retrieved_data == None:
            retrieved_data = self.db_retrieve_data(query=query)
            timestamp -= timedelta(days=1)
            query = 'SELECT timestamp, price FROM ' + table + ' WHERE timestamp = \'' + str(timestamp) + '\';'
        self.output = retrieved_data[0]
        return self.output

    """
    Helper function that formats a feature vector and returns it
    """
    def format_feature_vector(self):
        tmp_feature_vector = [item for sublist in self.feature_vector for item in sublist]
        self.feature_vector = tmp_feature_vector
        return self.feature_vector

    """
    Helper function that appends the sector data for the day to the current feature vector
    """
    def append_sector_info(self, table='SECTOR', timestamp=None):
        if timestamp is None:
            timestamp = self.start_date
        columns = 'timestamp, '
        for i, sector in enumerate(self.sectors):
            columns += sector + ', '
        columns = columns.strip(', ')
        query = 'SELECT ' + columns + ' FROM ' + table + ' WHERE timestamp < \'' + str(timestamp+timedelta(days=1)) + '\' AND timestamp > \'' + str(timestamp) + '\';'
        data = (self.db_retrieve_data(query, table='SECTOR'))
        if data is None:
            timestamp = datetime.date.today()
            query = 'SELECT ' + columns + ' FROM ' + table + ' WHERE timestamp < \'' + str(timestamp+timedelta(days=1)) + '\' AND timestamp > \'' + str(timestamp) + '\';'
            data = (self.db_retrieve_data(query, table='SECTOR'))
        new_data = [item for sublist in data for item in sublist]
        for datum in new_data:
            self.feature_vector.append(datum)

    """
    Helper function that dump a single feature vector
    """
    def dump_feature_vector(self, fname='sd_feature_vec', start_date=None, table=None, num_days=1):
        if start_date is not None and table is not None:
            if num_days != 1:
                with open('old/predictions/predictions_' + str(num_days) + '_days_' + str(start_date) + '.txt', 'w') as pred_f:
                    for i in range(num_days):
                        self.gen_feature_vector(table=table, start_date=start_date)
                        pred_f.write(str(i+1) + '.  ' + str(start_date + timedelta(days=self.days_out_prediction)) + ' - \n')
                        start_date -= timedelta(days=1)
                        with open('pickled_files/feature_vecs/' + fname + '_' + self.table + '_' + str(i) + '.pkl', 'wb') as f:
                            pickle.dump(self.feature_vector, f)
            start_date += timedelta(days=num_days)
            self.gen_feature_vector(table=table, start_date=start_date)
        with open('pickled_files/feature_vecs/' + fname + '_' + self.table + '.pkl', 'wb') as f:
            pickle.dump(self.feature_vector, f)

    """
    Helper function that loads in a single feature vector
    """
    def load_feature_vector(self, fname='sd_feature_vec'):
        with open('pickled_files/feature_vecs/' + fname + '_' + self.table + '.pkl', 'rb') as f:
            self.feature_vector = pickle.load(f)

    """
    Helper function that dumps the array of feature vectors stored in self.X
    """
    def dump_X(self, fname='sd_X'):
        with open('pickled_files/training_data/' + fname + '_' + self.table + '.pkl', 'wb') as f:
            pickle.dump([self.X, self.params], f)

    """
    Helper function that dumps the corresponding outputs stored in self.Y for the array of feature vectors
    """
    def dump_Y(self, fname='sd_Y'):
        with open('pickled_files/training_data/' + fname + '_' + self.table + '.pkl', 'wb') as f:
            pickle.dump(self.Y, f)

    """
    Helper function that loads stored versions of X
    """
    def load_X(self, fname='sd_X'):
        with open('pickled_files/training_data/' + fname + '_' + self.table + '.pkl', 'rb') as f:
            data = pickle.load(f)
            self.X, self.params = data[0], data[1]

    """
    Helper function that loads stored versions of Y
    """
    def load_Y(self, fname='sd_Y'):
        with open('pickled_files/training_data/' + fname + '_' + self.table + '.pkl', 'rb') as f:
            self.Y = pickle.load(f)

    """
    Helper function that loads the symbol for which feature vectors should be made for
    """
    def load_tables(self, fname='pickled_files/misc/symbols'):
        with open(fname + '.pkl', 'rb') as f:
            self.tables = pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description='Feature Vector options')
    parser.add_argument('--num_days', type=int, default=2500,
                        help="Number of days to make a feature vector for (length of X)")
    parser.add_argument('--step_size', type=int, default=1,
                        help="Step size of the days")
    args = parser.parse_args()

    # Load the symbols that are stored from the original stock_price_data.py script
    fname='pickled_files/misc/symbols'
    with open(fname + '.pkl', 'rb') as f:
        tables = pickle.load(f)
        for table in tables:
            # Initialize the model to generate each feature vector
            params = {}
            params['days_out_prediction'] = 7
            params['start_date'] = datetime.date.today()
            params['time_intervals_bool'] = True
            params['time_intervals'] = [1]
            params['sector_info'] = True
            params['table'] = table
            fv = FeatureVectorizor(params=params)

            # Cycle through the number of days at the given step size to make X and Y
            for i in range(0, args.num_days):
                feature_vector, output = fv.gen_feature_vector()
                fv.start_date -= timedelta(days=args.step_size)

            # Store the X and Y vectors into files
            fv.dump_X()
            fv.dump_Y()

    # Store the feature vector to perform a prediction on the data for the current day
    # start_date = datetime.date.today()
    # fv.dump_feature_vector(table='aapl', start_date=start_date, num_days=days_out_prediction)

if __name__ == '__main__':
    main()
