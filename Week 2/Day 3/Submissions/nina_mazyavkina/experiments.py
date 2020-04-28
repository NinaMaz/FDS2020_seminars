import timeit
from sklearn.ensemble import RandomForestRegressor

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