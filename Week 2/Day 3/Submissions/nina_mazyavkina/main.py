from __future__ import print_function
import argparse
from data_loader import DataLoader


def main(args):
    print("Setting up data directory")
    print("-------------------------")

    dl = DataLoader(args.data_path, args.url, args.rows)
    dl.data_pipeline()
    #random_array()
    #weather()

    print('Finished!')

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", help="the url to download the data from")
    parser.add_argument("-d", "--data_path", help="the path to store your data")
    parser.add_argument("-r", "--rows", type = int,  help="number of rows to use")
    return parser


if __name__ == '__main__':
    parser = arg_parser()   
    args = parser.parse_args()
    main(args)