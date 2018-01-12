import re
import os
import csv
import urllib
import pickle
import hashlib
import MySQLdb
import requests
import datetime
import numpy as np
import pandas as pd
from datetime import timedelta
from selenium import webdriver

class Crawl():
    def __init__(self, params, start_date=None, end_date=None):
        self.urls = ["https://news.google.com/news/?ned=us&gl=US&hl=en", "https://www.bing.com/news", "https://www.yahoo.com/news/"]
        self.header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}
        if start_date is None or end_date is None:
            self.start_date = datetime.date(2000, 1, 3)
            self.end_date = datetime.date.today()
        else:
            self.start_date = start_date
            self.end_date = end_date
        self.rng = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.stocks = []
        self.keywords = {}
        self.stocks_sa = {}
        self.df = pd.DataFrame(columns=['google_sa', 'bing_sa', 'yahoo_sa'], index=self.rng)
        self.params = params
        self.cur_sa = []
        self.cur_articles = []
        self.id = 0

    """
    Function to retrieve the news articles for a keyword, limit the news articles
    """
    def retrieve_articles(self, keyword, limit=10):
        # response = requests.get(url, headers=self.header)
        # content = response.content.decode('utf-8')
        #
        # matches = re.findall('<a href="([^ ]*)">(.*)<\/a>', content)
        # for match in matches:
        #     if (match is not None) & (len(match[1]) > 0):
        pass

    """
    Function to load the stocks located in csv files, also loads the first keyword
    """
    def load_stocks(self, fnames=['init_files/NASDAQ.csv', 'init_files/NYSE.csv']):
        for fname in fnames:
            with open(fname, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for i, row in enumerate(spamreader):
                    if i and len(row) > 1:
                        stock = row[0].replace(",", "").replace("\"", "")
                        keyword = row[1].replace(",", "").replace("\"", "")
                        self.stocks.append(stock)
                        self.stocks_sa[stock] = self.df
                        self.keywords[stock] = [keyword]
        return self.stocks

    """
    Function to dump the stocks sentiment analysis dictionary
    """
    def dump_stocks_sa(self, f_id=None):
        if f_id is None:
            pickle.dump(self.stocks_sa, open('pickled_files/news/stocks_sa_' + str(self.id) + '.pkl', "wb"))
            pickle.dump(self.params, open('pickled_files/news/stocks_sa_params_' + str(self.id) + '.pkl', "wb"))
        else:
            pickle.dump(self.stocks_sa, open('pickled_files/news/stocks_sa_' + str(f_id) + '.pkl', "wb"))
            pickle.dump(self.params, open('pickled_files/news/stocks_sa_params_' + str(f_id) + '.pkl', "wb"))

    """
    Function to load the stocks sentiment analysis dictionary
    """
    def load_stocks_sa(self, fname=None):
        if fname is None:
            self.stocks_sa = pickle.load(open('pickled_files/news/stocks_sa_' + str(self.id) + '.pkl', "rb"))
        else:
            self.stocks_sa = pickle.load(open(fname, "rb"))

    """
    Function to find the id of the sentiment analysis based on the parameters
    """
    def find_id(self):
        f_id = 0
        directory = "pickled_files/news/"
        for pfile in os.listdir(directory):
            if pfile.startswith("stocks_sa_params_"):
                f_id += 1
                cur_params = pickle.load(open(directory + pfile, "rb"))
                if cur_params == self.params:
                    self.id = f_id
                    return f_id
        self.id = f_id
        return f_id

def main():
        params = {'sa':False}
        newsCrawler = Crawl(params)
        stocks = newsCrawler.load_stocks()
        stocks = ['AAPL']
        for stock in stocks:
            keywords = newsCrawler.keywords[stock]
            for keyword in keywords:
                newsCrawler.retrieve_articles(keyword)
        newsCrawler.find_id()
        print(newsCrawler.id)
        newsCrawler.dump_stocks_sa()


if __name__ == '__main__':
    main()
