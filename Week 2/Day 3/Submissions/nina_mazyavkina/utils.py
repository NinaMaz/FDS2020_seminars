import pandas as pd

def results_printer(model_name, time, score):
	print("Computing Grid Search for {0}".format(model_name))
	print("Fitting time: {0}".format(time))
	print("R2 score: {0}".format(score))

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