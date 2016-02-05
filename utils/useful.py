from sklearn import tree
from sklearn.externals.six import StringIO
import os
import pydot

def PlotGraph(x_data,y_data,y_max_range,x_title,y_title,graph_name,plot_type):
    import matplotlib.pyplot as plt
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(graph_name)
    plt.plot(x_data,y_data,plot_type)
    plt.axis([0,max(x_data)+1,0,y_max_range])
    plt.savefig("experiments/pictures/"+graph_name+".png")
    plt.close()
    
def drawTree(data,X,y,name):
    clf = tree.DecisionTreeClassifier(criterion=data["criterion"],splitter=data["splitter"],max_features=data["max_features"],max_depth=data["max_depth"],min_samples_leaf=data["min_samples_leaf"],min_samples_split=data["min_samples_split"])
    clf = clf.fit(X,y)
    with open("iris.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
    os.unlink('iris.dot')
    dot_data = StringIO() 
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_pdf(name+".pdf")
    
    