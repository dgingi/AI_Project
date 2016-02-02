from FCSHC import FirstChoiceLocalSearch
from sklearn import tree
from pickle import dump,load
from utils.argumet_parsers import TestArgsParser
import datetime
import os
from exhandler.exhandler import EXHandler
from data.dbhandler import DBHandler
from utils.useful import PlotGraph
from utils.constants import LEAGUES
from data.cross_validation import CrossValidation

now = datetime.datetime.now()
LAST_YEAR = now.year - 1

args_parser = TestArgsParser()

def find_best_params():
    s = FirstChoiceLocalSearch()
    final_state, final_state_score, output_array = s.search()
    if not os.path.exists(os.path.join("tests/bprm")):
        os.makedirs("tests/bprm")
    with open(os.path.join("tests/bprm/bprm_fs1.pckl"),'w') as res:
        dump(final_state.data, res)
    with open(os.path.join("tests/bprm/bprm_fss1.pckl"),'w') as res:
        dump(final_state_score, res)
    with open(os.path.join("tests/bprm/bprm_oa1.pckl"),'w') as res:
        dump(output_array, res)

    y_data = [d[1] for d in output_array]
    PlotGraph([i for i in range(len(output_array))], y_data, 1, "iter number", "success rate", "best_params_test","k")
        
        
def find_best_lookback():
    with open("tests/best_params_test/best_params_test_fs.pckl",'r') as res:
        data = load(res)
    output_array = []
    for i in range(5,55,5):
        tot_res = 0.0
        for league in LEAGUES:
            E=EXHandler(league)
            X1,Y1,X2,Y2 = E.split_to_train_and_test(E.get())
            clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
            clf = clf.fit(X1,Y1)
            tot_res += E.predict(clf, X2, Y2)
        res = tot_res/len(LEAGUES)
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

def find_best_decision():
    with open("tests/bprm/bprm_fs1.pckl",'r') as res:
        data = load(res)
    cv = CrossValidation(test=False)
    cv.load_data(15)
    result_fix = 0.0
    for train , test in cv.leagues_cross_validation():
        clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
        clf = clf.fit(train[0],train[1])
        tags_arr = clf.predict(test[0])
        E=EXHandler("Primer_League",False)
        result_norm = E.predict(clf, test[0], test[1])    
        
        array = []
        for i in range(11):
            clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
            clf = clf.fit(train[0],train[1])
            res = clf.predict(test[0])
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
            if final[i]==test[1][i]:
                s+=1
                
                
        result_fix += (s*1.0)/len(final)
    print final,tags_arr,result_fix,result_norm

if __name__ == '__main__':
    args_parser.parse()
    run_func = args_parser.kwargs['func']
    run_func()
