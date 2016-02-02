from __future__ import print_function

import os
import subprocess

from time import time
from operator import itemgetter
from scipy.stats import randint

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import  cross_val_score
from data.cross_validation import CrossValidation
from exhandler.exhandler import EXHandler

def visualize_tree(tree, feature_names, fn="dt"):
    """Create tree png using graphviz.
    
    Args
    ----
    tree -- scikit-learn Decision Tree.
    feature_names -- list of feature names.
    fn -- [string], root of filename, default `dt`.
    """
    dotfile = fn + ".dot"
    pngfile = fn + ".png"

    with open(dotfile, 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)
    
    command = ["dot", "-Tpng", dotfile, "-o", pngfile]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, "
             "to produce visualization")

def get_iris_data():
    """Get the iris data, from local csv or pandas repo."""
    if os.path.exists("iris.csv"):
        print("-- iris.csv found locally")
        df = pd.read_csv("iris.csv", index_col=0)
    else:
        print("-- trying to download from github")
        fn = ("https://raw.githubusercontent.com/pydata/"
              "pandas/master/pandas/tests/data/iris.csv")
        try:
            df = pd.read_csv(fn)
        except:
            exit("-- Unable to download iris.csv")

        with open("iris.csv", 'w') as f:
            print("-- writing to local iris.csv file")
            df.to_csv(f)

    return df

def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.
    
    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters

def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.
    
    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return  top_params

def run_randomsearch(X, y, clf, para_dist, cv=5,
                     n_iter_search=20):
    """Run a random search for best Decision Tree parameters.
    
    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_dist -- [dict] list, distributions of parameters
                  to sample
    cv -- fold of cross-validation, default 5
    n_iter_search -- number of random parameter sets to try,
                     default 20.

    Returns
    -------
    top_params -- [dict] from report()
    """
    random_search = RandomizedSearchCV(clf, 
                    	param_distributions=param_dist,
                        n_iter=n_iter_search)
    
    start = time()
    random_search.fit(X, y)
    print(("\nRandomizedSearchCV took {:.2f} seconds "
           "for {:d} candidates parameter "
           "settings.").format((time() - start),
                               n_iter_search))

    top_params = report(random_search.grid_scores_, 3)
    return  top_params


if __name__ == "__main__":

    cv = CrossValidation(test=False)
    cv.load_data(15)
    
    E = EXHandler("Primer_League",False)
    features_names = E.get_features_names()
    X = cv.complete_examples
    y = cv.complete_tags
    
    dt_old = DecisionTreeClassifier()
    dt_old.fit(X, y)
    scores = cross_val_score(dt_old, X, y, cv=cv.leagues_cross_validation)
    print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), 
                                              scores.std()),
                                              end="\n\n" )
    
    # set of parameters to test
    param_grid = {"criterion": ["gini", "entropy"],
                  "min_samples_split": [i+1 for i in range(len(500))],
                  "max_depth": [None]+[i+1 for i in range(len(59))],
                  "min_samples_leaf": [i+1 for i in range(len(200))],
                  "max_leaf_nodes": [None]+[i+1 for i in range(1,len(400))],
                  }
    
    dt = DecisionTreeClassifier()
    ts_gs = run_gridsearch(X, y, dt, param_grid, cv=cv.leagues_cross_validation)
    
    print("\n-- Best Parameters:")
    for k, v in ts_gs.items():
        print("parameter: {:<20s} setting: {}".format(k, v))
    
    # test the retuned best parameters
    print("\n\n-- Testing best parameters [Grid]...")
    dt_ts_gs = DecisionTreeClassifier(**ts_gs)
    scores = cross_val_score(dt_ts_gs, X, y, cv=cv.leagues_cross_validation)
    print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), 
                                              scores.std()),
                                              end="\n\n" )
    
    
    visualize_tree(dt_ts_gs, features_names, fn="grid_best")
    
    print("-- Random Parameter Search via 10-fold CV")
    
    # dict of parameter list/distributions to sample
    param_dist = {"criterion": ["gini", "entropy"],
                  "min_samples_split": randint(1, 500),
                  "max_depth": randint(1, 59),
                  "min_samples_leaf": randint(1, 200),
                  "max_leaf_nodes": randint(2, 400)}
    
    dt = DecisionTreeClassifier()
    ts_rs = run_randomsearch(X, y, dt, param_dist, cv=cv.leagues_cross_validation,
                             n_iter_search=288)
    
    print("\n-- Best Parameters:")
    for k, v in ts_rs.items():
        print("parameters: {:<20s} setting: {}".format(k, v))
    
    # test the retuned best parameters
    print("\n\n-- Testing best parameters [Random]...")
    dt_ts_rs = DecisionTreeClassifier(**ts_rs)
    scores = cross_val_score(dt_ts_rs, X, y, cv=10)
    print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), 
                                              scores.std()),
                                              end="\n\n" )
    
    visualize_tree(dt_ts_rs, features_names, fn="rand_best")
