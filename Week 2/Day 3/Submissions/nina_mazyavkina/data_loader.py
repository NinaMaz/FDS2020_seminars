from __future__ import print_function
import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
from glob import glob


class DataLoader:
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