from old_utils import *
from FCSHC import *
from pickle import dump
from sklearn import tree


def find_best_params():
    E = EXHandler("Primer_League")
    ex, ta = E.get()
    X1,X2,Y1,Y2 = ex[:1409],ex[1409:],ta[:1409],ta[1409:]
    s = FirstChoiceLocalSearch(X1,Y1)
    final_state, final_state_score, output_array = s.search(X2, Y2)
    with open("tests/best_params_test_fs.pckl",'w') as res:
        dump(final_state.data, res)
    with open("tests/best_params_test_fss.pckl",'w') as res:
        dump(final_state_score, res)
    with open("tests/best_params_test_oa.pckl",'w') as res:
        dump(output_array, res)
    y_data = [d[1] for d in output_array]
    PlotGraph([i for i in range(len(output_array))], y_data, 1, "iter number", "success rate", "best_params_test","k")
        
def find_best_partition():
    E = EXHandler("Primer_League")
    ex, ta = E.get()
    X1,X2,Y1,Y2 = ex[:1409],ex[1409:],ta[:1409],ta[1409:]
    best_i = 0
    with open("tests/best_params_test_fs.pckl",'r') as res:
        data = load(res)
    clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
    clf = clf.fit(X1,Y1)
    y_data = []
    best_res = E.predict(clf, X2, Y2)
    y_data += [(best_i,best_res)]
    for i in range(2,20):
        new_X1 = E.convert(X1, i)
        new_X2 = E.convert(X2, i)
        clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
        clf = clf.fit(new_X1,Y1)
        res = E.predict(clf, new_X2, Y2)
        y_data += [(i,res)]
    with open("tests/best_partition.pckl",'w') as res:
        dump(y_data, res)
    PlotGraph([i for i in range(21)], y_data, "partition size", "success rate", "best_partition", "ro")
        
        