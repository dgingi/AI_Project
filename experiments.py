from data.cross_validation import CrossValidation
from os.path import exists, join as join_path
from os import  makedirs
from pickle import dump , load
import numpy as np
from scipy.stats import ttest_rel
from glob import glob
from sklearn.cross_validation import  cross_val_score
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as AdaC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

class Experiment():
    '''
    Abstract Experiment class.
    
    Experiments classes provide basic functionality to running an experiment-
            allows saving and loading the results
            automatically loads data for cross validation
    '''
    def __init__(self,dir_name,test=False):
        '''
        Creates a new Experiment instance.
        
        Requires a dir_name (to save results) and a flag to indicate whether it's test environment or not.
        '''
        self._dir_name = dir_name
        self.results_dir = join_path('Results',dir_name)
        self._test = test
        
        self._loaded_data = None
        
    def save(self,data):
        '''
        Saves data into the results dir under the special format {experiment_name}.results
        '''
        if not exists(self.results_dir):
            makedirs(self.results_dir)
        with open(join_path(self.results_dir,self.name+'.results'),'w') as _f:
            dump(data, _f)
        
    def load(self):
        '''
        Loads results from previous runs into _loaded_data attribute.
        '''
        _path = glob(join_path(self.results_dir,'%s.results'%self.name)).pop()
        with open(_path,'r') as _f:
            self._loaded_data = load(_f)
    
    def get_data(self):
        '''
        Loads all the examples and tags needed for the experiment.
        
        Loads the all of the examples and tags, and also creates a cross validation for using in the estimators \ searches.
        '''
        self.cv = CrossValidation(test=self._test)
        lookback = 2 if self._test else 15
        self.cv.load_data(lookback)
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags
        
    def run(self):
        '''
        Runs the experiment.
        
        You should override this function in derived classes to configure the experiment
        '''
        self.get_data()
        
    def load_params(self):
        '''
        Loads the parameters of the experiment.
        
        You should override this in derived classes to configure the experiment parameters
        '''
        raise NotImplementedError
    
    '''
    @todo:    t_test for 2 algs
              validate & report
    '''
      
    def t_test(self,original_measurements, measurements_after_alteration):
        '''
        This is the paired T-test (for repeated measurements):
        Given two sets of measurements on the SAME data points (folds) before and after some change,
        Checks whether or not they come from the same distribution. 
        
        This T-test assumes the measurements come from normal distributions.
        
        Returns: 
            The probability the measurements come the same distributions
            A flag that indicates whether the result is statically significant
            A flag that indicates whether the new measurements are better than the old measurements
        '''
        SIGNIFICANCE_THRESHOLD= 0.05
        
        test_value, probability= ttest_rel(original_measurements, measurements_after_alteration)
        is_significant= probability/2 < SIGNIFICANCE_THRESHOLD
        is_better= sum(original_measurements) < sum(measurements_after_alteration) #should actually compare averages, but there's no need since it's the same number of measurments.
        return probability/2 if is_better else 1-probability/2, is_significant, is_better
    
    def _load_prev_experiment(self,exp):
        try:
            exp.load()
        except Exception as e:
            print 'Failed to load previous %s experiment\n. If you would like to run the %s experiment, Please type:\n Yes I am sure'%exp.name
            ans = raw_input('>>>')
            if ans == 'Yes I am sure':
                exp.run()
            else:
                return
  
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
    '''
    An Experiment class to search for the best hyperparameters for an estimator.
    
    The search is done using RandomGridSearchCV, on the DecisionTreeClassifer estimator and the 
    RandomForestClassifier estimator. 
    '''
    def __init__(self,dir_name,test=False):
        Experiment.__init__(self,dir_name,test)
        self.name = 'Best_Params'
        
    def load_params(self):
        '''
        Loads the parameters for the experiment - the grids to search on.
        
        To change parameters or values for one of the estimators - change to correct field.
        '''
        if not self._test:
            return {'DTC':{'criterion':['gini','entropy'],\
                           'max_depth':range(5,61,5),\
                           'max_leaf_nodes':[None]+range(10,51,5),\
                           'min_samples_leaf':range(15,100,15),\
                           'min_samples_split':range(2,51,2),\
                           'splitter':['random','best'],\
                           'max_features':[None,'auto','log2']+range(10,61,5)},\
                    'RFC':{'criterion':['gini','entropy'],\
                           'max_depth':range(5,61,5),\
                            'max_leaf_nodes':[None]+range(10,51,5),\
                            'min_samples_leaf':range(15,100,15),\
                            'min_samples_split':range(2,51,2),\
                            'n_estimators':range(50,401,50),\
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
        '''
        Runs a RandomizedSearch on both DecisionTree and RandomForest classifiers.
        '''
        Experiment.run(self)
        _grids = self.load_params()
        grid_tree = RandomizedSearchCV(DTC(), _grids['DTC'], n_jobs=-1, cv=self.cv.leagues_cross_validation,n_iter=1000)
        grid_tree.fit(self.cv.complete_examples,self.cv.complete_tags)
        grid_forest = RandomizedSearchCV(RFC(), _grids['RFC'], n_jobs=-1, cv=self.cv.leagues_cross_validation,n_iter=1000)
        grid_forest.fit(self.cv.complete_examples,self.cv.complete_tags)
        self._loaded_data = {'Tree':grid_tree,'Forest':grid_forest} 
        self.save(self._loaded_data)
        

class AdaBoostExperimet(Experiment):
    '''
    A class the experiments the AdaBoost algorithm.
    '''
    def __init__(self, dir_name, test=False):
        Experiment.__init__(self, dir_name, test=test)
        self.name = 'AdaBoost'
        
    def load_params(self,estimators=[]):
        if not self._test:
            return {'base_estimator':estimators,\
                'n_estimators':range(100,501,50),\
                }
        else:
            return {'base_estimator':estimators,\
                    'n_estimators':range(50,151,50)}
            
    def run(self):
        Experiment.run(self)
        best_param_exp = BestParamsExperiment(self._dir_name, self._test)
        self._load_prev_experiment(best_param_exp)
        estimators = [DTC(**best_param_exp._loaded_data['Tree'].best_params_),RFC(**best_param_exp._loaded_data['Forest'].best_params_)]
        _grid = self.load_params(estimators)
        ada_boost = GridSearchCV(AdaC(), _grid, n_jobs=-1, cv=self.cv.leagues_cross_validation)
        ada_boost.fit(self.cv.complete_examples,self.cv.complete_tags)
        self._loaded_data = {'AdaBoost':ada_boost}
        self.save(self._loaded_data)
        
class BestLookbackExperimet(Experiment):
    '''
    A class that experiments the best lookback for making examples.
    '''
    def __init__(self, dir_name, test=False):
        Experiment.__init__(self, dir_name, test=test)
        self.name = 'Best_Lookback'
             
    
    def load_params(self,estimators=[]):
        best_param_exp = BestParamsExperiment(self._dir_name, self._test)
        self._load_prev_experiment(best_param_exp)
        ada_exp = AdaBoostExperimet(self._dir_name, self._test)
        self._load_prev_experiment(ada_exp)
        self.estimators = [DTC(**best_param_exp._loaded_data['Tree'].best_params_),\
                           RFC(**best_param_exp._loaded_data['Forest'].best_params_),\
                           AdaC(**ada_exp._loaded_data['AdaBoost'].best_params_)]
        
    def get_data(self,lookback):
        self.cv = CrossValidation(test=self._test)
        self.cv.load_data(lookback)
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags
        
    def run(self):
        if not self._test:
            self.ranges = [15,30]
        else:
            self.ranges = range(1,101,10)
        self.load_params()
        results = {str(i):0 for i in self.ranges}
        for lookback in self.ranges:
            self.get_data(lookback)
            dtc_score = cross_val_score(self.estimators[0], self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
            rfc_score = cross_val_score(self.estimators[1], self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
            ada_score = cross_val_score(self.estimators[2], self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
            results[str(lookback)] = (dtc_score,rfc_score,ada_score)
        self._loaded_data = results
        self.save(self._loaded_data)
        
        
if __name__ == '__main__':
    AdaBoostExperimet('Best_Params').run()
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
    
