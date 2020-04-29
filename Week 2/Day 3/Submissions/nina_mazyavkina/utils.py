import pandas as pd
import csv
import os
import joblib
from dask.distributed import Client
from distributed.deploy.local import LocalCluster
from sklearn.externals import joblib

def file_is_empty(path):
    return os.stat(path).st_size==0

def save_to_file(path, dict1):
    header = list(dict1.keys())
    with open(path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if file_is_empty(path):
            writer.writeheader()
        writer.writerow(dict1)

def results_printer(model_name, time, score, csv_path = None):
    print("Computing Grid Search for {0}".format(model_name))
    print("Fitting time: {0}".format(time))
    print("R2 score: {0}".format(score))
    if csv_path:
        save_to_file(csv_path, {'model_name':model_name, 'time': time, 'score':score})

def preprocess(filenames, labels_to_drop):
    dfs = []
    for fn in filenames:
        df = pd.read_csv(fn)
        
        df = df.drop(labels_to_drop, 1)
        df = df.dropna(0)
        #encoding
        obj_df = df.select_dtypes(include=['object']).columns
        obj_clmns = df[obj_df].astype('category').copy()
        for i in obj_df:
            df[i] = obj_clmns[i].cat.codes
        dfs.append(df)
        del df
    return pd.concat(dfs, sort = False, ignore_index = True)

def setup_local_cluster(n_workers = 4):
    cluster = LocalCluster(n_workers)
    client = Client(cluster)
    return client