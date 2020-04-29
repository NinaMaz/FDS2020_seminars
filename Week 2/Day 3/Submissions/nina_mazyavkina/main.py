from __future__ import print_function
import argparse
import timeit
import os
from data_processing import DataLoader, NycFlightsData
from sklearn.model_selection import train_test_split
from utils import results_printer, preprocess, setup_local_cluster
from experiments import Experiment
from glob import glob
from sklearn.ensemble import RandomForestRegressor
from dask_ml.xgboost import XGBRegressor


def main(args):
    """The Experimental pipeline:
        1) line 32: create the data handling object of the class NycFlightsData
        2) lines 42, 62: create the experiment handling object of the class Experiment
        3) lines 45, 63: call .run_gs method, which runs the GridSearchCV
        4) lines 49, 67: fit the models with the best found parameters using .fit_model method
        The experiments' outputs are printed into the terminal and, optionally, to the .csv file, specified at lines 52, 67 
        in the results_printer method. 
    """
    print("Setting up data directory")
    print("-------------------------")

    dl = DataLoader(args.data_path, args.url, args.rows)
    dl.data_pipeline()
    print('Finished!')


    client = setup_local_cluster()
    client.shutdown()

    filenames = sorted(glob(os.path.join(args.data_path,'nycflights', '1990.csv')))
    labels_to_drop = ['Year','DepTime', 'CRSDepTime','ArrTime', 'AirTime','CRSArrTime','CRSElapsedTime', 'TailNum', 'TaxiIn','TaxiOut']

    nycdata = NycFlightsData(filenames,labels_to_drop)

    #RFRegressor

    param_grid = {
    'max_depth': list(range(2,10,2)),
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': [100, 300],
    
    }
    exp = Experiment(nycdata)
    model = RandomForestRegressor(max_depth = 2, random_state=0)
    best_est, time = exp.run_gs(model, param_grid, cluster = True)
    results_printer('RandomForestRegressor GS', time, None, args.csv)

    time, score = exp.fit_model(model)
    results_printer('RandomForestRegressor', time, score, args.csv)

    #DaskXGboost



    param_grid = {
    'max_depth': list(range(2,10,2)),
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [100, 300],    
    }

    exp = Experiment(nycdata, use_dask = True, n_partitions = 4)
    model = XGBRegressor(max_depth = 2, random_state=0)
    best_est, time = exp.run_gs(model, param_grid, cluster = False)
    results_printer('DaskXGBRegressor GS', time, None, args.csv)

    time, score = exp.fit_model(model)
    results_printer('DaskXGBRegressor', time, score, args.csv)


    #random_array()
    #weather()

    

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", help="the url to download the data from")
    parser.add_argument("-d", "--data_path", help="the path to store your data")
    parser.add_argument("-r", "--rows", type = int,  help="number of rows to use")
    parser.add_argument("-c", "--csv", help="file in which to store logs")

    return parser


if __name__ == '__main__':
    parser = arg_parser()   
    args = parser.parse_args()
    main(args)