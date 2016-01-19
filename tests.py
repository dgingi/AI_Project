from old_utils import *
from FCSHC import *
from pickle import dump
from sklearn import tree
from utils.argumet_parsers import TestArgsParser
import os
import datetime

now = datetime.datetime.now()
LAST_YEAR = now.year - 1

args_parser = TestArgsParser()

def get_examples_and_tags(league):
    E = EXHandler(league)
    X , Y = E.get()
    X2, Y2 = E.get(LAST_YEAR)
    idx = -1*len(X2)
    X1,Y1 = X[:idx], Y[:idx]
    return X1,Y1,X2,Y2

def find_best_params(league):
    X1,Y1,X2,Y2 = get_examples_and_tags(league)
    s = FirstChoiceLocalSearch(X1,Y1)
    final_state, final_state_score, output_array, random_array = s.search(X2, Y2)
    with open(os.path.join("tests/best_params_test/best_params_test_fs.pckl"),'w') as res:
        dump(final_state.data, res)
    with open(os.path.join("tests/best_params_test/best_params_test_fss.pckl"),'w') as res:
        dump(final_state_score, res)
    with open(os.path.join("tests/best_params_test/best_params_test_oa.pckl"),'w') as res:
        dump(output_array, res)
    y_data = [d[1] for d in output_array]
    PlotGraph([i for i in range(len(output_array))], y_data, 1, "iter number", "success rate", "best_params_test","k")
        
def find_best_partition(league):
    E = EXHandler(league)
    X1,Y1,X2,Y2 = get_examples_and_tags(league)
    best_i = 0
    
    s = FirstChoiceLocalSearch(X1,Y1)
    data, final_state_score, output_array = s.search(X2, Y2)

    clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
    clf = clf.fit(X1,Y1)
    output_array = []
    best_res = E.predict(clf, X2, Y2)
    output_array += [(best_i,best_res)]
    
    for i in range(2,10):
        new_X1 = E.convert(X1, i)
        new_X2 = E.convert(X2, i)
        s = FirstChoiceLocalSearch(new_X1,Y1)
        data, final_state_score, output_array = s.search(new_X2, Y2)
        clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
        clf = clf.fit(new_X1,Y1)
        res = E.predict(clf, new_X2, Y2)
        output_array += [(i,res)]
    with open("tests/best_partition_test/best_partition.pckl",'w') as res:
        dump(output_array, res)
    x_data = [d[0] for d in output_array]
    y_data = [d[1] for d in output_array]
    PlotGraph(x_data, y_data, 1, "partition size", "success rate", "best_partition", "k")
    
def find_best_lookback(league):
    D = DBHandler(league)
    E = EXHandler(league)
    with open("tests/best_params_test/best_params_test_fs.pckl",'r') as res:
        data = load(res)
    output_array = []
    for i in range(3,16):
        ex_10 , ta_10 = D.create_examples("2010", i)
        ex_11 , ta_11 = D.create_examples("2011", i)
        ex_12 , ta_12 = D.create_examples("2012", i)
        ex_13 , ta_13 = D.create_examples("2013", i)
        ex_14 , ta_14 = D.create_examples("2014", i)
        X1 = ex_10 + ex_11 + ex_12 + ex_13
        Y1 = ta_10 + ta_11 + ta_12 + ta_13
        clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
        clf = clf.fit(X1,Y1)
        res = E.predict(clf, ex_14, ta_14)
        output_array += [(i,res)]
    with open("tests/best_lookback_test/best_lookback.pckl",'w') as res:
        dump(output_array, res)
    x_data = [d[0] for d in output_array]
    y_data = [d[1] for d in output_array]
    PlotGraph(x_data, y_data, 1, "lookback size", "success rate", "best_lookback", "k")

def find_best_lookback_and_params(league):
    D = DBHandler(league)
    E = EXHandler(league)
    output_array = []
    for i in [3,5,7,10,15,20]:
        ex_10 , ta_10 = D.create_examples("2011", i)
        ex_11 , ta_11 = D.create_examples("2011", i)
        ex_12 , ta_12 = D.create_examples("2012", i)
        ex_13 , ta_13 = D.create_examples("2013", i)
        ex_14 , ta_14 = D.create_examples("2014", i)
        X1 = ex_10 + ex_11 + ex_12 + ex_13
        Y1 = ta_10 + ta_11 + ta_12 + ta_13
        s = FirstChoiceLocalSearch(X1,Y1)
        data, final_state_score, output_array = s.search(ex_14, ta_14)
        clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
        clf = clf.fit(X1,Y1)
        res = E.predict(clf, ex_14, ta_14)
        output_array += [(i,res)]
    with open("tests/best_lookback_and_params_test/best_lookback_and_params.pckl",'w') as res:
        dump(output_array, res)
    x_data = [d[0] for d in output_array]
    y_data = [d[1] for d in output_array]
    PlotGraph(x_data, y_data, 1, "lookback size", "success rate", "best_lookback_and_params", "k")

def find_best_decision(X1,X2,Y1,Y2):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X1,Y1)
    tags_arr = E.predict(X2)
    E=EXHandler("Primer_League")
    result_norm = E.predict(clf, X2, Y2)    
    
    array = []
    for i in range(19):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X1,Y1)
        res = clf.predict(X2)
        array += [res]
    
    final = []
    for i in range(len(array[0])):
        temp = []
        for j in range(len(array)):
            temp += [array[j][i]]
        d = {-1:0,0:0,1:0}
        for k in temp:
            d[k]+=1
        max_list = []
        for key in d:
            max_list += [(d[key],key)]
        final += [max(max_list)[1]]
    
    s=0
    for i in range(len(final)):
        if final[i]==Y2[i]:
            s+=1
            
            
    result_fix = (s*1.0)/len(final)
    return final,tags_arr,result_fix,result_norm

if __name__ == '__main__':
    args_parser.parse()
    run_func = args_parser.kwargs['func']
    run_func(args_parser.kwargs['league'])
