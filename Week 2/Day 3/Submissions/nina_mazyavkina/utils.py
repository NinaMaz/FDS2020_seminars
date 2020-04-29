import pandas as pd
import csv
import os
import joblib
from dask.distributed import Client
from distributed.deploy.local import LocalCluster
from sklearn.externals import joblib

def file_is_empty(path):
    """Checking the existance of a file"""
    return os.stat(path).st_size==0

def save_to_file(path, dict1):
    """Saving the dataframe to a file, 
       if it doesn't exist, first create the file"""
    header = list(dict1.keys())
    with open(path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if file_is_empty(path):
            writer.writeheader()
        writer.writerow(dict1)

def results_printer(model_name, time, score, csv_path = None):
    """Printing the results,
       if csv_path is set to some path - write the results down in a .csv file"""
    print("Computing {0}".format(model_name))
    print("Fitting time: {0}".format(time))
    print("R2 score: {0}".format(score))
    if csv_path:
        save_to_file(csv_path, {'model_name':model_name, 'time': time, 'score':score})

def setup_local_cluster(n_workers = 4):
    """Cluster setter"""
    cluster = LocalCluster(n_workers)
    client = Client(cluster)
    return client