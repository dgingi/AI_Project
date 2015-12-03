from old_utils import *
from FCSHC import *
from pickle import dump
from sklearn import tree


def find_best_params(X1,X2,Y1,Y2,psize):
    s = FirstChoiceLocalSearch(X1,Y1)
    final_state, final_state_score, output_array = s.search(X2, Y2)
    with open("tests/best_params_test/"+psize+"_best_params_test_fs.pckl",'w') as res:
        dump(final_state.data, res)
    with open("tests/best_params_test/"+psize+"_best_params_test_fss.pckl",'w') as res:
        dump(final_state_score, res)
    with open("tests/best_params_test/"+psize+"_best_params_test_oa.pckl",'w') as res:
        dump(output_array, res)
    y_data = [d[1] for d in output_array]
    PlotGraph([i for i in range(len(output_array))], y_data, 1, "iter number", "success rate", psize+"_best_params_test","k")
        
def find_best_partition():
    E = EXHandler("Primer_League")
    ex_11 , ta_11 = E.get(2011)
    ex_12 , ta_12 = E.get(2012)
    ex_13 , ta_13 = E.get(2013)
    ex_14 , ta_14 = E.get(2014)
    X1,Y1 = ex_11+ex_12+ex_13, ta_11+ta_12+ta_13
    X2,Y2 = ex_14, ta_14
    best_i = 0
    find_best_params(X1, X2, Y1, Y2, "0")
    with open("tests/best_params_test/0_best_params_test_fs.pckl",'r') as res:
        data = load(res)
    clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
    clf = clf.fit(X1,Y1)
    output_array = []
    best_res = E.predict(clf, X2, Y2)
    output_array += [(best_i,best_res)]
    for i in range(2,10):
        new_X1 = E.convert(X1, i)
        new_X2 = E.convert(X2, i)
        print "start best param for",i
        find_best_params(new_X1, new_X2, Y1, Y2, str(i))
        with open("tests/best_params_test/"+str(i)+"_best_params_test_fs.pckl",'r') as res:
            data = load(res)
        clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
        clf = clf.fit(new_X1,Y1)
        res = E.predict(clf, new_X2, Y2)
        output_array += [(i,res)]
    with open("tests/best_partition_test/best_partition.pckl",'w') as res:
        dump(output_array, res)
    x_data = [d[0] for d in output_array]
    y_data = [d[1] for d in output_array]
    PlotGraph(x_data, y_data, 1, "partition size", "success rate", "best_partition", "k")
    
def find_best_lookback():
    D = DBHandler("Primer_League")
    E = EXHandler("Primer_League")
    with open("tests/best_params_test/0_best_params_test_fs.pckl",'r') as res:
        data = load(res)
    output_array = []
    for i in range(3,16):
        ex_11 , ta_11 = D.create_examples("2011", i)
        ex_12 , ta_12 = D.create_examples("2012", i)
        ex_13 , ta_13 = D.create_examples("2013", i)
        ex_14 , ta_14 = D.create_examples("2014", i)
        X1 = ex_11 + ex_12 + ex_13
        Y1 = ta_11 + ta_12 + ta_13
        clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
        clf = clf.fit(X1,Y1)
        res = E.predict(clf, ex_14, ta_14)
        output_array += [(i,res)]
    with open("tests/best_lookback_test/best_lookback.pckl",'w') as res:
        dump(output_array, res)
    x_data = [d[0] for d in output_array]
    y_data = [d[1] for d in output_array]
    PlotGraph(x_data, y_data, 1, "lookback size", "success rate", "best_lookback", "k")

def find_best_lookback_and_params():
    D = DBHandler("Primer_League")
    E = EXHandler("Primer_League")
    output_array = []
    for i in [3,5,7,10,15,20]:
        ex_11 , ta_11 = D.create_examples("2011", i)
        ex_12 , ta_12 = D.create_examples("2012", i)
        ex_13 , ta_13 = D.create_examples("2013", i)
        ex_14 , ta_14 = D.create_examples("2014", i)
        X1 = ex_11 + ex_12 + ex_13
        Y1 = ta_11 + ta_12 + ta_13
        find_best_params(X1, ex_14, Y1, ta_14, "0L")
        with open("tests/best_params_test/0L_best_params_test_fs.pckl",'r') as res:
            data = load(res)
        clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
        clf = clf.fit(X1,Y1)
        res = E.predict(clf, ex_14, ta_14)
        output_array += [(i,res)]
    with open("tests/best_lookback_and_params_test/best_lookback_and_params.pckl",'w') as res:
        dump(output_array, res)
    x_data = [d[0] for d in output_array]
    y_data = [d[1] for d in output_array]
    PlotGraph(x_data, y_data, 1, "lookback size", "success rate", "best_lookback_and_params", "k")


    