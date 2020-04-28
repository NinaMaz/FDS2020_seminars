import os
import pandas as pd

def results_printer(model_name, time, score):
	print("Computing Grid Search for {0}".format(model_name))
	print("Fitting time: {0}".format(time))
	print("R2 score: {0}".format(score))

def preprocess(file_name, labels_to_drop):
    df = pd.read_csv(os.path.join('nycflights', file_name))
    
    df = df.dropna(1, 'all').dropna(0)
    df = df.drop(labels_to_drop, 1)
    
    #encoding
    obj_df = df.select_dtypes(include=['object']).columns
    obj_clmns = df[obj_df].astype('category').copy()
    for i in obj_df:
        df[i] = obj_clmns[i].cat.codes
    
    return df
