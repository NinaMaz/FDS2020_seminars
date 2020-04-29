from __future__ import print_function
import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
import dask.dataframe as dd
from glob import glob
from sklearn.model_selection import train_test_split



class DataLoader:
    """Handles loading the data from url into data_dir"""
    def __init__(self, data_dir, url, num_rows):
        self.data_dir = data_dir
        self.url = url
        self.num_rows = num_rows

    def load_data(self):
        flights_raw = os.path.join(self.data_dir, 'nycflights.tar.gz')
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        if not os.path.exists(flights_raw):
            print("- Downloading NYC Flights dataset... ", end='', flush=True)
            urllib.request.urlretrieve(self.url, flights_raw)
            print("done", flush=True)

    def extract_data(self):       
        flightdir = os.path.join(self.data_dir, 'nycflights')
        if not os.path.exists(flightdir):
            print("- Extracting flight data... ", end='', flush=True)
            tar_path = os.path.join(self.data_dir, 'nycflights.tar.gz')
            with tarfile.open(tar_path, mode='r:gz') as flights:
                flights.extractall(self.data_dir)
            print("done", flush=True)

    def create_json(self):
        jsondir = os.path.join(self.data_dir, 'flightjson')
        if not os.path.exists(jsondir):
            print("- Creating json data... ", end='', flush=True)
            os.mkdir(jsondir)
            for path in glob(os.path.join(self.data_dir, 'nycflights', '*.csv')):
                prefix = os.path.splitext(os.path.basename(path))[0]
                # Just take the first num_rows rows for the demo
                df = pd.read_csv(path).iloc[:self.num_rows]
                df.to_json(os.path.join(self.data_dir, 'flightjson', prefix + '.json'),
                           orient='records', lines=True)
            print("done", flush=True)

    def data_pipeline(self):
        self.load_data()
        self.extract_data()
        self.create_json()

class NycFlightsData():
    """Class providing the methods for working with the raw data
    ---
    labels_to_drop (list) - the features to drop from the resulting dataframe
    """
    def __init__(self, filenames, labels_to_drop):
        self.filenames = filenames
        self.labels_to_drop = labels_to_drop

    def tt_split(self, df, y_label = 'DepDelay', test_size = 0.33):
        X = df.drop(y_label, 1)
        y = df[y_label]
        split = train_test_split(X, y, test_size=test_size, random_state=42)
        return split

    def preprocess(self):
        dfs = []
        for fn in self.filenames:
            df = pd.read_csv(fn)
            
            df = df.drop(self.labels_to_drop, 1)
            df = df.dropna(0)

            #encoding
            obj_df = df.select_dtypes(include=['object']).columns
            obj_clmns = df[obj_df].astype('category').copy()
            for i in obj_df:
                df[i] = obj_clmns[i].cat.codes
            dfs.append(df)
            del df
        return pd.concat(dfs, sort = False, ignore_index = True)
        
    def to_dask_array(self, df, n_partitions = 4):
        df = dd.from_pandas(df, npartitions=4).to_dask_array(lengths=True)
        return df
    def from_dask_array(self, df):
        df = df.compute()
        return df