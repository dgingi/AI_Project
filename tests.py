from old_utils import *
from FCSHC import *
from pickle import dump
from sklearn import tree


def find_best_params(X1,X2,Y1,Y2,psize):
    ''''E = EXHandler("Primer_League")
    ex, ta = E.get()
    X1,X2,Y1,Y2 = ex[:1409],ex[1409:],ta[:1409],ta[1409:]'''
    s = FirstChoiceLocalSearch(X1,Y1)
    final_state, final_state_score, output_array = s.search(X2, Y2)
    with open("tests/"+psize+"_best_params_test_fs.pckl",'w') as res:
        dump(final_state.data, res)
    with open("tests/"+psize+"_best_params_test_fss.pckl",'w') as res:
        dump(final_state_score, res)
    with open("tests/"+psize+"_best_params_test_oa.pckl",'w') as res:
        dump(output_array, res)
    y_data = [d[1] for d in output_array]
    PlotGraph([i for i in range(len(output_array))], y_data, 1, "iter number", "success rate", psize+"_best_params_test","k")
        
def find_best_partition():
    E = EXHandler("Primer_League")
    ex, ta = E.get()
    X1,X2,Y1,Y2 = ex[:1409],ex[1409:],ta[:1409],ta[1409:]
    best_i = 0
    find_best_params(X1, X2, Y1, Y2, "0")
    with open("tests/0_best_params_test_fs.pckl",'r') as res:
        data = load(res)
    clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
    clf = clf.fit(X1,Y1)
    output_array = []
    best_res = E.predict(clf, X2, Y2)
    output_array += [(best_i,best_res)]
    for i in range(2,20):
        new_X1 = E.convert(X1, i)
        new_X2 = E.convert(X2, i)
        find_best_params(new_X1, new_X2, Y1, Y2, str(i))
        with open("tests/"+str(i)+"_best_params_test_fs.pckl",'r') as res:
            data = load(res)
        clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
        clf = clf.fit(new_X1,Y1)
        res = E.predict(clf, new_X2, Y2)
        output_array += [(i,res)]
    with open("tests/best_partition.pckl",'w') as res:
        dump(output_array, res)
    y_data = [d[1] for d in output_array]
    x_data = [d[0] for d in output_array]
    PlotGraph(x_data, y_data, 1, "partition size", "success rate", "best_partition", "k")
        
        