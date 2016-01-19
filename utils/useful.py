
def PlotGraph(x_data,y_data,y_max_range,x_title,y_title,graph_name,plot_type):
    import matplotlib.pyplot as plt
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(graph_name)
    plt.plot(x_data,y_data,plot_type)
    plt.axis([0,max(x_data)+1,0,y_max_range])
    plt.savefig("tests/pictures/"+graph_name+".png")
    plt.close()