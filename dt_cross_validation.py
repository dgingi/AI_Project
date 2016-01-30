#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.
"""dt_cross_validation.py -- use cross-validation to choose best decision
   tree parameters.
"""

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

def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    """Produce pseudo-code for decision tree.
    
    Args
    ----
    tree -- scikit-leant Decision Tree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
   
    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse (left, right, threshold, features,
                             left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse (left, right, threshold, features,
                             right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")
    
    recurse(left, right, threshold, features, 0, 0)


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


def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas Data Frame.
    target_column -- column to map to int, producing new
                     Target column.

    Returns
    -------
    df -- modified Data Frame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


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
    print("\n-- get data:")
    df = get_iris_data()
    print("")
    
    features = ["SepalLength", "SepalWidth",
                "PetalLength", "PetalWidth"]
    df, targets = encode_target(df, "Name")
    from pickle import load
    with open("tests\\X.pckl",'r') as res:
        X=load(res)
    with open("tests\\Y.pckl",'r') as res:
        y=load(res)
    #y = df["Target"]
    #X = df[features]
    
    print("-- 10-fold cross-validation "
          "[using setup from previous post]")
    dt_old = DecisionTreeClassifier(min_samples_split=20,
                                    random_state=99)
    dt_old.fit(X, y)
    scores = cross_val_score(dt_old, X, y, cv=10)
    print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), 
                                              scores.std()),
                                              end="\n\n" )
    
    print("-- Grid Parameter Search via 10-fold CV")
    
    # set of parameters to test
    param_grid = {"criterion": ["gini", "entropy"],
                  "min_samples_split": [2, 10, 20],
                  "max_depth": [None, 2, 5, 10],
                  "min_samples_leaf": [1, 5, 10],
                  "max_leaf_nodes": [None, 5, 10, 20],
                  }
    
    dt = DecisionTreeClassifier()
    ts_gs = run_gridsearch(X, y, dt, param_grid, cv=10)
    
    print("\n-- Best Parameters:")
    for k, v in ts_gs.items():
        print("parameter: {:<20s} setting: {}".format(k, v))
    
    # test the retuned best parameters
    print("\n\n-- Testing best parameters [Grid]...")
    dt_ts_gs = DecisionTreeClassifier(**ts_gs)
    scores = cross_val_score(dt_ts_gs, X, y, cv=10)
    print("mean: {:.3f} (std: {:.3f})".format(scores.mean(), 
                                              scores.std()),
                                              end="\n\n" )
    
    print("\n-- get_code for best parameters [Grid]:", end="\n\n")
    dt_ts_gs.fit(X,y)
    get_code(dt_ts_gs, features, targets)
    
    visualize_tree(dt_ts_gs, features, fn="grid_best")
    
    print("-- Random Parameter Search via 10-fold CV")
    
    # dict of parameter list/distributions to sample
    param_dist = {"criterion": ["gini", "entropy"],
                  "min_samples_split": randint(1, 20),
                  "max_depth": randint(1, 20),
                  "min_samples_leaf": randint(1, 20),
                  "max_leaf_nodes": randint(2, 20)}
    
    dt = DecisionTreeClassifier()
    ts_rs = run_randomsearch(X, y, dt, param_dist, cv=10,
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
    
    print("\n-- get_code for best parameters [Random]:")
    dt_ts_rs.fit(X,y)
    get_code(dt_ts_rs, features, targets)
    
    visualize_tree(dt_ts_rs, features, fn="rand_best")
