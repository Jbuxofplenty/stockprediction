import pickle
import urllib
import argparse
import datetime
import requests
from datetime import timedelta

class FeatureVectorizor():
    """
    Class that provides a framework for creating arrays of feature vectors that can be used to train machine learning models
    """
    def __init__(self, params):
        # Variables to load in the stock price data for a particular stock and the sector info
        self.sp_data = {}
        self.sector_info = {}

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
        self.sectors = ['information_technology', 'health_care', 'materials', 'financials', 'consumer_discretionary', 'industrials', 'consumer_staples', 'utilities', 'real_estate', 'energy', 'telecommunication_services']
        self.table = self.params['table']

    """
    Function that generates a single feature vector and appends it to X (the array of feature vecs for a model to be trained on)
    """
    def gen_feature_vector(self, start_date=None):
        # Reinitialize all of the variables
        self.load_sp_data()
        self.load_sector_info()
        self.feature_vector = []
        self.data = []
        self.output = -1
        if start_date is not None:
            self.start_date = start_date

        # Cycle through each of the time intervals to generate a list of parameters to be added to the feature vector
        retrieved_data = None
        timestamp = str(self.start_date)
        while retrieved_data is None:
            retrieved_data = self.format_data(timestamp=timestamp.split()[0])
            timestamp = str(datetime.datetime.strptime(timestamp.split()[0], '%Y-%m-%d') - timedelta(days=1))
        self.feature_vector = retrieved_data
        if self.params['sector_info']:
            self.append_sector_info(timestamp=timestamp.split()[0])
        self.find_output(table=self.table, time_interval=self.days_out_prediction)
        self.X[str(self.start_date)] = self.feature_vector
        self.Y[str(self.start_date)] = self.output
        print(str(self.start_date), self.feature_vector[0])
        return self.feature_vector, self.output

    """
    Helper function to format the data from the sp_data pickled dict to a list for use in the feature vector
    """
    def format_data(self, timestamp):
        try:
            data = []
            for type in self.sp_data.keys():
                if self.params['type_options'][type]:
                    for key in self.sp_data[type][timestamp].keys():
                        data.append(self.sp_data[type][timestamp][key])
            return data
        except:
            return None

    """
    Helper function that finds the correct output to append to self.Y given a time interval, date, and table
    """
    def find_output(self, start_date=None, time_interval=7, table=None):
        if start_date is None:
            start_date = self.start_date
        timestamp = str(start_date + timedelta(days=time_interval))
        retrieved_data = None
        while retrieved_data is None:
            try:
                retrieved_data = self.sp_data['TIME_SERIES_DAILY_ADJUSTED'][timestamp.split()[0]]['adjusted close']
            except:
                retrieved_data = None
            timestamp = str(datetime.datetime.strptime(timestamp.split()[0], '%Y-%m-%d') - timedelta(days=1))
        self.output = retrieved_data

    """
    Helper function that appends the sector data for the day to the current feature vector
    """
    def append_sector_info(self, table='SECTOR', timestamp=None):
        if timestamp is None:
            timestamp = str(self.start_date).split()[0]
        try:
            data = []
            for time_period in self.sector_info[timestamp].keys():
                for key in self.sector_info[timestamp][time_period].keys():
                    self.feature_vector.append(float(self.sector_info[timestamp][time_period][key].strip('%'))/100)
        except KeyError:
            return


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
        with open('pickled_files/training_data/' + str(self.table[0]) + '/' + fname + '_' + self.table + '.pkl', 'wb') as f:
            pickle.dump([self.X, self.params], f)

    """
    Helper function that dumps the corresponding outputs stored in self.Y for the array of feature vectors
    """
    def dump_Y(self, fname='sd_Y'):
        with open('pickled_files/training_data/' + str(self.table[0]) + '/' + fname + '_' + self.table + '.pkl', 'wb') as f:
            pickle.dump(self.Y, f)

    """
    Helper function that loads stored versions of X
    """
    def load_X(self, fname='sd_X'):
        with open('pickled_files/training_data/' + str(self.table[0]) + '/' + fname + '_' + self.table + '.pkl', 'rb') as f:
            data = pickle.load(f)
            self.X, self.params = data[0], data[1]

    """
    Helper function that loads stored versions of Y
    """
    def load_Y(self, fname='sd_Y'):
        with open('pickled_files/training_data/' + str(self.table[0]) + '/' + fname + '_' + self.table + '.pkl', 'rb') as f:
            self.Y = pickle.load(f)

    """
    Helper function that loads the symbol for which feature vectors should be made for
    """
    def load_tables(self, fname='pickled_files/symbols/symbols'):
        with open(fname + '.pkl', 'rb') as f:
            self.tables = pickle.load(f)

    """
    Helper function that loads the stock data stored in self.sp_data to a dict
    """
    def load_sp_data(self, fname=None):
        if fname is None:
            with open('pickled_files/sp_data/' + str(self.table[0]) + '/' + str(self.table) + '.pkl', 'rb') as f:
                self.sp_data = pickle.load(f)
        else:
            fname = 'sp_data'
            with open('pickled_files/sp_data/' + fname + '.pkl', 'rb') as f:
                self.sp_data = pickle.load(f)

    """
    Helper function that loads the sector information into self.sector_info
    """
    def load_sector_info(self):
        with open('pickled_files/sp_data/s/SECTOR.pkl', 'rb') as f:
            self.sector_info = pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description='Feature Vector options')
    parser.add_argument('--num_days', type=int, default=2500,
                        help="Number of days to make a feature vector for (length of X)")
    parser.add_argument('--step_size', type=int, default=1,
                        help="Step size of the days")
    args = parser.parse_args()

    # Load the symbols that are stored from the original stock_price_data.py script
    fname='pickled_files/symbols/symbols'
    with open(fname + '.pkl', 'rb') as f:
        tables = pickle.load(f)
        tables = ['aapl']
        for table in tables:
            # Initialize the model to generate each feature vector
            params = {}
            params['days_out_prediction'] = 7
            params['start_date'] = datetime.date.today()
            params['sector_info'] = False
            params['type_options'] = {'STOCH':True, 'MACD':True, 'RSI':True, 'EMA':True, 'SMA':True, 'TIME_SERIES_DAILY_ADJUSTED':True}
            params['table'] = table
            fv = FeatureVectorizor(params=params)
            fv.load_sp_data()
            print(table, ':', len(fv.sp_data['TIME_SERIES_DAILY_ADJUSTED']))
            if len(fv.sp_data['TIME_SERIES_DAILY_ADJUSTED']) < args.num_days:
                num_days = len(fv.sp_data['TIME_SERIES_DAILY_ADJUSTED'])
            else:
                num_days = args.num_days

            # Cycle through the number of days at the given step size to make X and Y
            for i in range(0, num_days):
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
