from utils.argumet_parsers import TestArgsParser
import datetime
from data.cross_validation import CrossValidation
from operator import itemgetter
import numpy as np
from sklearn.cross_validation import  cross_val_score

now = datetime.datetime.now()
LAST_YEAR = now.year - 1

args_parser = TestArgsParser()

class run_experiment():
    def __init__(self,dir_name,algo,experiment):
        self.dir_name = dir_name
        self.algo = algo
        self.expr = experiment
    
    def get_data(self):
        self.cv = CrossValidation(test=True)
        self.cv.load_data()
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags
        
    def report(self,grid_scores, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1),reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print(("Mean validation score: "
                   "{0:.3f} (std: {1:.3f})").format(
                   score.mean_validation_score,
                   np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")
    
        return top_scores[0].parameters

class best_params(run_experiment):
    def __init__(self,dir_name,algo,experiment,params,criterions,params_ranges):
        run_experiment.__init__(self,dir_name,algo,experiment)
        self.params = params
        self.criterion = criterions
        self.params_ranges = params_ranges
        self.algo = algo
        self.expr = experiment
    
    def make_param_grid(self):
        self.param_grid = {param:[] for param in self.params}
        self.param_grid["criterion"] = self.criterion
        for param in self.params:
            if param != "criterion":
                self.param_grid[param] = [i for i in range(self.params_ranges[param][0],self.params_ranges[param][1])]
        
    def run(self):
        self.get_data()
        algo_old = self.algo()
        self.make_param_grid()
        algo_search = self.expr(algo_old, self.param_grid, cv=5)
        algo_search.fit(self.X, self.y)
        tg_pg = self.report(self.param_grid.grid_scores_, 3)
        new_algo = self.algo(**tg_pg)
        scores = cross_val_score(new_algo, self.X, self.y, cv=self.cv.leagues_cross_validation)
        print "mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std())+"\n\n" 
        
        
        
        
if __name__ == '__main__':
    #args_parser.parse()
    #run_func = args_parser.kwargs['func']
    #run_func()
    '''
    Example how to run best params with grid search
    '''
    from sklearn.grid_search import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier as DTC
    parameters = ["criterion","min_samples_split","max_depth","min_samples_leaf","max_leaf_nodes"]
    criterion = ["gini","entropy"]
    params_ranges = {"min_samples_split": (1,300),
                  "max_depth": (1,60),
                  "min_samples_leaf": (15,100),
                  "max_leaf_nodes": (2,100),
                  }
    expr = best_params("bprm_grid",DTC,GridSearchCV,parameters,criterion,params_ranges)
    expr.run()
    
