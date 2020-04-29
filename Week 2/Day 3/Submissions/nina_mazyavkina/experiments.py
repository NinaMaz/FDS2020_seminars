import timeit
import pandas as pd
import joblib
from utils import setup_local_cluster
from sklearn.model_selection import GridSearchCV


def dask_experiment():

    starttime = timeit.default_timer()
    ###
    ###
    e_time = timeit.default_timer() - starttime


    return

class Experiment():
    def __init__(self, nycdata, use_dask = False, n_partitions = 4):
        self.use_dask = use_dask
        self.X_train, self.X_test, self.y_train, self.y_test = self.collect_data(nycdata, n_partitions)
        

    def collect_data(self, nycdata, n_partitions):       
        split = nycdata.tt_split(nycdata.preprocess())
        if self.use_dask:
            split = [nycdata.to_dask_array(s, n_partitions = n_partitions) for s in split] 
        return split

    def run_gs(self, model, param_grid, cv = 3, cluster = True):
        grid_search = GridSearchCV(model, param_grid, verbose=2, cv=cv, n_jobs=-1)
        if cluster:
            client = setup_local_cluster()
            starttime = timeit.default_timer()
            with joblib.parallel_backend("dask", scatter=[self.X_train, self.y_train]):
                grid_search.fit(self.X_train, self.y_train)
            e_time = timeit.default_timer() - starttime
            client.shutdown()
        else:
           starttime = timeit.default_timer()
           grid_search.fit(self.X_train, self.y_train)
           e_time = timeit.default_timer() - starttime       

        return grid_search.best_estimator_, e_time


    def fit_model(self, model):
        if self.use_dask:
            client = setup_local_cluster()
            starttime = timeit.default_timer()
            model.fit(self.X_train, self.y_train)
            e_time = timeit.default_timer() - starttime
            score = model.score(self.X_test, self.y_test)
            client.shutdown()
        else:
            starttime = timeit.default_timer()
            model.fit(self.X_train, self.y_train)
            e_time = timeit.default_timer() - starttime
            score = model.score(self.X_test, self.y_test)

        return e_time, score



        
