import timeit
import pandas as pd
import joblib
from utils import setup_local_cluster
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



def sk_experiment(X_train, X_test, y_train, y_test):
    #TO DO add GridSearch
    model = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)
    starttime = timeit.default_timer()
    model.fit(X_train, y_train)
    e_time = timeit.default_timer() - starttime
    score = model.score(X_test, y_test)
    return e_time, score

def dask_experiment():

    starttime = timeit.default_timer()
    ###
    ###
    e_time = timeit.default_timer() - starttime


    return

class NycFlightsData():
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

class Experiment():
    def __init__(self, nycdata, cluster = True):

        self.X_train, self.X_test, self.y_train, self.y_test = self.collect_data(nycdata)

    def collect_data(self, nycdata):        
        split = nycdata.tt_split(nycdata.preprocess())
        return split

    def run_gs(self, model, param_grid):
        if cluster:
            setup_local_cluster()
        grid_search = GridSearchCV(model, param_grid, verbose=2, cv=3, n_jobs=-1)

        starttime = timeit.default_timer()
        with joblib.parallel_backend("dask", scatter=[self.X_train, self.y_train]):
            grid_search.fit(self.X_train, self.y_train)
        e_time = timeit.default_timer() - starttime

        return grid_search.best_params_, e_time


    def fit_model(self, model, params):
        starttime = timeit.default_timer()
        model.fit(self.X_train, self.y_train)
        e_time = timeit.default_timer() - starttime
        score = model.score(self.X_test, self.y_test)
        return e_time, score



        
