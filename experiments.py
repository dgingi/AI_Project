
from data.cross_validation import CrossValidation
from operator import itemgetter
import numpy as np
from sklearn.cross_validation import  cross_val_score
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.grid_search import GridSearchCV

class Experiment():
    def __init__(self,dir_name,test=False):
        self.results_dir = dir_name
        self._test = test
        
    def save(self):
        raise NotImplemented
    
    def get_data(self):
        self.cv = CrossValidation(test=self._test)
        self.cv.load_data()
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags
        
    def run(self):
        self.get_data()
        
    def load_params(self):
        raise NotImplementedError
        
#     def report(self,grid_scores, n_top=3):
#         top_scores = sorted(grid_scores, key=itemgetter(1),reverse=True)[:n_top]
#         for i, score in enumerate(top_scores):
#             print("Model with rank: {0}".format(i + 1))
#             print(("Mean validation score: "
#                    "{0:.3f} (std: {1:.3f})").format(
#                    score.mean_validation_score,
#                    np.std(score.cv_validation_scores)))
#             print("Parameters: {0}".format(score.parameters))
#             print("")
#     
#         return top_scores[0].parameters

class BestParamsExperiment(Experiment):
    def __init__(self,dir_name,test=False):
        Experiment.__init__(self,dir_name,test)
        
    
    def make_param_grid(self):
        self.param_grid = {param:[] for param in self.params}
        self.param_grid["criterion"] = self.criterion
        for param in self.params:
            if param != "criterion":
                self.param_grid[param] = [i for i in range(self.params_ranges[param][0],self.params_ranges[param][1])]
        
    def load_params(self):
        if not self._test:
            return {'DTC':{'criterion':['gini','entropy'],\
                           'max_depth':range(5,61,5),\
                           'max_leaf_nodes':[None]+range(10,51,5),\
                           'min_samples_leaf':range(15,100,15),\
                           'min_samples_split':range(2,51),\
                           'splitter':['random','best'],\
                           'max_features':[None,'auto','log2']+range(10,61,5)},\
                    'RFC':{'criterion':['gini','entropy'],\
                           'max_depth':range(5,61,5),\
                            'max_leaf_nodes':[None]+range(10,51,5),\
                            'min_samples_leaf':range(15,100,15),\
                            'min_samples_split':range(2,51,2),\
                            'n_estimators':range(50,400,50),\
                            'max_features':[None,'auto','log2']+range(10,61,5),\
                           'n_jobs':[-1]}}
        else:
            return {'DTC':{'criterion':['gini'],\
#                            'max_depth':range(5,61,30),\
#                            'max_leaf_nodes':[None]+range(10,51,30),\
#                            'min_samples_leaf':range(15,100,60),\
#                            'min_samples_split':range(2,51,30),\
#                            'splitter':['random','best'],\
                           'max_features':[None,'auto','log2']+range(10,61,25)},\
                    'RFC':[{'criterion':['gini'],\
#                            'max_depth':range(5,61,30),\
#                            'max_leaf_nodes':[None]+range(10,51,30),\
#                            'min_samples_leaf':range(15,100,60),\
#                            'min_samples_split':range(2,51,30),\
#                            'n_estimators':range(50,400,150),\
                           'max_features':[None,'auto','log2']+range(10,61,25),\
                           'n_jobs':[-1]}]}
    
    def run(self):
        Experiment.run(self)
        _grids = self.load_params()
        grid_tree = GridSearchCV(DTC(), _grids['DTC'], n_jobs=-1, cv=self.cv.leagues_cross_validation)
        grid_tree.fit(self.cv.complete_examples,self.cv.complete_tags)
        grid_forest = GridSearchCV(RFC(), _grids['RFC'], n_jobs=-1, cv=self.cv.leagues_cross_validation)
        grid_forest.fit(self.cv.complete_examples,self.cv.complete_tags)
        self.save()
        
        
        
# if __name__ == '__main__':
#     #args_parser.parse()
#     #run_func = args_parser.kwargs['func']
#     #run_func()
#     '''
#     Example how to run best params with grid search
#     '''
#     from sklearn.grid_search import GridSearchCV
#     from sklearn.tree import DecisionTreeClassifier as DTC
#     parameters = ["criterion","min_samples_split","max_depth","min_samples_leaf","max_leaf_nodes"]
#     criterion = ["gini","entropy"]
#     params_ranges = {"min_samples_split": (1,300),
#                   "max_depth": (1,60),
#                   "min_samples_leaf": (15,100),
#                   "max_leaf_nodes": (2,100),
#                   }
#     expr = best_params("bprm_grid",DTC,GridSearchCV,parameters,criterion,params_ranges)
#     expr.run()
    
