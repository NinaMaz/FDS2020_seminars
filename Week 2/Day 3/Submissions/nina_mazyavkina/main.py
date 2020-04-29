from __future__ import print_function
import argparse
import timeit
import os
from data_loader import DataLoader
from sklearn.model_selection import train_test_split
from utils import results_printer, preprocess
from experiments import sk_experiment
from glob import glob
from experiments import NycFlightsData


def main(args):
    print("Setting up data directory")
    print("-------------------------")

    dl = DataLoader(args.data_path, args.url, args.rows)
    dl.data_pipeline()
    print('Finished!')

    #TO DO: this should go into the Experiment class
    filenames = sorted(glob(os.path.join(args.data_path,'nycflights', '1990.csv')))
    labels_to_drop = ['Year','DepTime', 'CRSDepTime','ArrTime', 'AirTime','CRSArrTime','CRSElapsedTime', 'TailNum', 'TaxiIn','TaxiOut']
    # df = preprocess(filenames, labels_to_drop)

    # X = df.drop('DepDelay', 1)
    # y = df['DepDelay']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    nycdata = NycFlightsData(filenames,labels_to_drop)

    X_train, X_test, y_train, y_test = nycdata.tt_split(nycdata.preprocess())

    time, score = sk_experiment(X_train, X_test, y_train, y_test)
    results_printer('RandomForestRegressor', time, score, args.csv)

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