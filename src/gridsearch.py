#!/usr/bin/env python3

import operator

from copy import deepcopy
from sklearn.model_selection import GridSearchCV

def timeit(f):
    import time
    def wrapper(*args, **kwargs):
        s = time.time()
        r = f(*args, **kwargs)
        e = time.time()
        print(f'Took {e-s} seconds.')
        return r
    return wrapper

@timeit
def grid_search(X, y, parameters, n_jobs=-1):
    """
    Grid search looking for best estimator for each model candidate
    Args:
        X: Features
        y: labels
        parameters: parameters dict
        n_jobs: number of concurrently running workers, default: -1
    Return:
        result: dict containing best params sorted by best_score
    """

    results = []

    for params in deepcopy(parameters):

        clf = params['clf'][0]
        params.pop('clf')
        grid = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1)
        grid.fit(X, y)

        results.append\
        (
            {
                'grid': grid,
                'classifier': grid.best_estimator_,
                'best_params': grid.best_params_,
                'best_score': grid.best_score_,
                'std_test_score': grid.cv_results_['std_test_score'][grid.best_index_]
            }
        )

    return sorted(results, key=operator.itemgetter('best_score'), reverse=True)
