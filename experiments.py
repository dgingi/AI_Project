from glob import glob
from os import  makedirs
from os.path import exists, join as join_path
from pickle import dump , load
from scipy.stats import ttest_rel
from sklearn.cross_validation import  cross_val_score
from sklearn.ensemble import AdaBoostClassifier as AdaC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as DTC
from tabulate import tabulate
import warnings

from data.cross_validation import CrossValidation
from utils.argumet_parsers import ExperimentArgsParser
from utils.decorators import timed
from utils.constants import LEAGUES, MAX_YEAR

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
    @timed    
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
        '''
        A function to load a previous experiment.
        '''
        try:
            exp.load()
        except Exception as e:
            print 'Failed to load previous {ex} experiment\n. If you would like to run the {ex} experiment, Please type:\n Yes I am sure'.format(ex=exp.name)
            ans = raw_input('>>>')
            if ans == 'Yes I am sure':
                exp.run()
            else:
                return False
  
    def report(self,verbosity,outfile):
        '''
        A method to report the experiment's results.
        '''
        if self._loaded_data is None:
            try:
                self.load()
            except:
                if not self._load_prev_experiment(self):
                    print 'Can not report, must run experiment before!'
                    return
        print self._begining_report
        if verbosity == 0:
            print self._no_detail
        elif verbosity == 1:
            print self._detail
        elif verbosity == 2:
            print self._more_detail
        print self._ending_report
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
                           'max_features':[None,'auto','log2']+range(10,61,25)},\
                    'RFC':[{'criterion':['gini'],\
                           'max_features':[None,'auto','log2']+range(10,61,25),\
                           'n_jobs':[-1]}]}
    
    
    _begining_report = '''This experiment performed a Randomized Search for 1000 iterations upon the hyper parameters grid for both the \
Decision Tree classifier and the Random Forest classifier.'''
            
    _ending_report = '''Done'''
    
    @property        
    def _no_detail(self):
        '''
        Reporting on low verbosity
        '''
        return '\n'.join(['Decision Tree after search accuracy score: {0:.4f}'.format(self._loaded_data['Tree'].best_score_),\
                          str(self._loaded_data['Tree'].best_params_),
        'Random Forest after search accuracy score: {0:.4f}'.format(self._loaded_data['Forest'].best_score_),\
        str(self._loaded_data['Forest'].best_params_)])
    @timed    
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
        
class BayesExperiment(Experiment):
    def __init__(self,dir_name,test=False):
        Experiment.__init__(self,dir_name,test)
        self.name = 'Bayes'
        
    def run(self):
        Experiment.run(self)
        bayes_score = cross_val_score(GaussianNB(), self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        self._loaded_data = {'Bayes':bayes_score}
        self.save(self._loaded_data)
        
    _begining_report = '''This experiment tried a Naive Bayes classifer.'''
            
    _ending_report = '''Done'''
    
    @property        
    def _no_detail(self):
        '''
        Reporting on low verbosity
        '''
        return '\n'.join(['Gaussian Naive Bayes accuracy score: {0:.4f}'.format(self._loaded_data['Bayes'].mean())])
        
    
class LearningCurveExperiment(Experiment):
    def __init__(self,dir_name,test=False):
        Experiment.__init__(self,dir_name,test)
        self.name = 'Learning_Curve'
        
    def run(self):
        Experiment.run(self)
        best_param_exp = BestParamsExperiment(self._dir_name, self._test)
        self._load_prev_experiment(best_param_exp)
        tree_curve = learning_curve(DTC(**best_param_exp._loaded_data['Tree'].best_params_), self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        forest_curve = learning_curve(RFC(**best_param_exp._loaded_data['Forest'].best_params_), self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        bayes_curve = learning_curve(GaussianNB(), self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        self._loaded_data = {'Tree_Curve':tree_curve,'Forest_Curve':forest_curve,'Bayes_Curve':bayes_curve}
        self.save(self._loaded_data)
        
    _begining_report = '''This experiment checks the learning curve for all the classifiers. \n
Will plot the learning curves on screen.'''
            
    _ending_report = '''Done'''
    
    @property        
    def _no_detail(self):
        '''
        Reporting on low verbosity
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import numpy as np
            import matplotlib.pyplot as plt
    
    
            def plot_learning_curve(title, ylim=None,_type='Tree'):
          
                plt.figure()
                plt.title(title)
                if ylim is not None:
                    plt.ylim(*ylim)
                plt.xlabel("Training examples")
                plt.ylabel("Score")
                train_sizes, train_scores, test_scores = self._loaded_data[_type]
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                plt.grid()
            
                plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1,
                                 color="r")
                plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
                plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Training score")
                plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Cross-validation score")
            
                plt.legend(loc="best")
                return plt
                    
            plot_learning_curve("Learning Curves (Decision Tree)",_type='Tree_Curve')
            plot_learning_curve("Learning Curves (Random Forest)",_type='Forest_Curve')
            plot_learning_curve("Learning Curves (Naive Bayes)",_type='Bayes_Curve')
            
            plt.show()
            return ''
        
        
        
    
        
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
        estimators = [DTC(),RFC(),DTC(**best_param_exp._loaded_data['Tree'].best_params_),RFC(**best_param_exp._loaded_data['Forest'].best_params_)]
        _grid = self.load_params(estimators)
        ada_boost = GridSearchCV(AdaC(), _grid, n_jobs=-1, cv=self.cv.leagues_cross_validation)
        ada_boost.fit(self.cv.complete_examples,self.cv.complete_tags)
        self._loaded_data = {'AdaBoost':ada_boost}
        self.save(self._loaded_data)
        
    _begining_report = '''This experiment performed a Grid Search upon the hyper parameters grid for the \
AdaBoost classifier.'''
            
    _ending_report = '''Done'''
    
    @property        
    def _no_detail(self):
        '''
        Reporting on low verbosity
        '''
        return '\n'.join(['AdaBoost after search accuracy score: {0:.4f}'.format(self._loaded_data['AdaBoost'].best_score_),\
                          str(self._loaded_data['AdaBoost'].best_params_)])
        
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
#         ada_exp = AdaBoostExperimet(self._dir_name, self._test)
#         self._load_prev_experiment(ada_exp)
        self.estimators = [DTC(**best_param_exp._loaded_data['Tree'].best_params_),\
                           RFC(**best_param_exp._loaded_data['Forest'].best_params_)]
#                            AdaC(**ada_exp._loaded_data['AdaBoost'].best_params_)]
        
    def get_data(self,lookback):
        self.cv = CrossValidation(test=self._test,remote=self._remote)
        self.cv.load_data(lookback)
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags
        
    def run(self):
        if self._test:
            self.ranges = [15,30]
        else:
            self.ranges = range(1,70,5)
        self.load_params()
        results = {str(i):0 for i in self.ranges}
        self._remote = True
        for lookback in self.ranges:
            self.get_data(lookback)
            self._remote = False
            dtc_score = cross_val_score(self.estimators[0], self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
            rfc_score = cross_val_score(self.estimators[1], self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
#             ada_score = cross_val_score(self.estimators[2], self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
            results[str(lookback)] = (dtc_score,rfc_score)
        self._loaded_data = results
        self.save(self._loaded_data)
        

    _begining_report = '''This experiment checks both classifier's accuracy correlation with the lookback parameter \
for the creation on the examples.'''
            
    _ending_report = '''Done'''
    
    @property        
    def _no_detail(self):
        '''
        Reporting on low verbosity
        '''
        _dtc_scores = {int(_lk):self._loaded_data[_lk][0].mean() for _lk in self._loaded_data}
        _rfc_scores = {int(_lk):self._loaded_data[_lk][1].mean() for _lk in self._loaded_data}
        _table = tabulate([['Decision Tree']+[value for (key, value) in sorted(_dtc_scores.items())],\
                           ['Random Forest']+[value for (key, value) in sorted(_rfc_scores.items())]],\
                          headers=['Classifier / Lookback']+sorted(_dtc_scores),tablefmt="fancy_grid",floatfmt=".4f")
        return 'Cross validation scores for each classifier by lookback:\n%s\n'%_table

class BestForestSizeExperiment(Experiment):
    '''
    A class that experiments the best size of a non-random forest.
    All trees are the same tree but decision tree gives different result each time you do fit so this experoment checks
    the best size of a forest.
    '''
    def __init__(self, dir_name, test=False):
        Experiment.__init__(self, dir_name, test=test)
        self.name = 'Best_Forest_Size'
    
    def load_params(self):
        best_param_exp = BestParamsExperiment("Best_Params", self._test)
        try:
            best_param_exp.load()
        except Exception as e:
            print 'Failed to load previous %s experiment\n. If you would like to run the %s experiment, Please type:\n Yes I am sure'
            ans = raw_input('>>>')
            if ans == 'Yes I am sure':
                best_param_exp.run()
            else:
                return
        self.estimators_params = {'DTC':best_param_exp._loaded_data['Tree'].best_params_,'RTC':best_param_exp._loaded_data['Forest'].best_params_}
    
    def run(self):
        Experiment.run(self)
        self.load_params()
        if self._test:
            self.ranges = [1,3]
        else:
            self.ranges = [1,3,5,7,9,11,13,15]
        self._loaded_data = {k:0 for k in self.ranges}
        
        for _range in self.ranges:
            cross_size = 0
            decision_result = 0.0
            
            for train , test in self.cv._leagues_cross_validation():  
                cross_size += 1
                
                tags_array = []
                for i in range(_range):
                    clf = DTC(**self.estimators_params['DTC'])
                    clf = clf.fit(train[0],train[1])
                    res_tags = clf.predict(test[0])
                    tags_array += [res_tags]
                
                final_decsion = []
                for i in range(len(tags_array[0])):
                    temp = []
                    for j in range(_range):
                        temp += [tags_array[j][i]]
                    decision_dict = {-1:0,0:0,1:0}
                    for res in temp:
                        decision_dict[res]+=1
                    
                    max_list = []
                    for key in decision_dict:
                        max_list += [(decision_dict[key],key)]
                    final_decsion += [max(max_list)[1]]
                
                score = 0
                for i in range(len(final_decsion)):
                    if final_decsion[i] == test[1][i]:
                        score+=1
                        
                decision_result += (score*1.0)/len(final_decsion)
            decision_result /= cross_size
            self._loaded_data[_range] = decision_result
        self.save(self._loaded_data)
        
    _begining_report = '''This experiment checks the best size of a non-random forest. \
Due to decision-tree behavior we build a forest from the same tree and the result of an example will be \
determined by max result from all trees.'''
            
    _ending_report = '''Done'''
    
    @property        
    def _no_detail(self):
        '''
        Reporting on low verbosity
        '''
        _tree_size_scores = {int(_k):self._loaded_data[_k] for _k in self._loaded_data.keys()}
        _table = tabulate([['Forest Size']+[value for (key, value) in sorted(_tree_size_scores.items())]],\
                          headers=['Experiment / Size']+sorted(_tree_size_scores),tablefmt="fancy_grid",floatfmt=".4f")
        return 'Cross validation scores for each non-random forest size :\n%s\n'%_table
  
class BestProbaForDecision(Experiment):
    '''
    A class that experiments the best proba from which we want to make the decision.
    '''
    def __init__(self, dir_name, test=False):
        Experiment.__init__(self, dir_name, test=test)
        self.name = 'Best_Proba'
    
    def load_params(self):
        best_param_exp = BestParamsExperiment("Best_Params", self._test)
        try:
            best_param_exp.load()
        except Exception as e:
            print 'Failed to load previous %s experiment\n. If you would like to run the %s experiment, Please type:\n Yes I am sure'
            ans = raw_input('>>>')
            if ans == 'Yes I am sure':
                best_param_exp.run()
            else:
                return
        self.estimators_params = {'DTC':best_param_exp._loaded_data['Tree'].best_params_,'RTC':best_param_exp._loaded_data['Forest'].best_params_}
    
    def run(self):
        Experiment.run(self)
        self.load_params()
        if self._test:
            self.ranges = [0.34,0.35]
        else:
            self.ranges = [(float(_i)/100) for _i in range(34,60)]
        self._loaded_data = {k:(0,0) for k in self.ranges}
        
        for _range in self.ranges:
            decision_result = 0.0
            score = 0
            curr_decisions = 0
            
            for train , test in self.cv._leagues_cross_validation():  
                clf = DTC(**self.estimators_params['DTC'])
                clf = clf.fit(train[0],train[1])
                res_tags = clf.predict(test[0])
                res_proba = clf.predict_proba(test[0])
                    
                for i in range(len(res_tags)):
                    if max(res_proba[i]) >= _range:
                        curr_decisions += 1
                        if res_tags[i] == test[1][i]:
                            score += 1
                        
            decision_result = (score*1.0)/curr_decisions
            self._loaded_data[_range] = (curr_decisions,decision_result)
        self.save(self._loaded_data)
        
    _begining_report = '''This experiment checks the best probability given by the Decision Tree from which  \
we start making the decisions.'''
            
    _ending_report = '''Done'''
    
    @property        
    def _no_detail(self):
        '''
        Reporting on low verbosity
        '''
        _proba_scores = {float(_k):self._loaded_data[_k] for _k in self._loaded_data.keys()}
        _inner_table = [[key,value[0],value[1]] for (key, value) in sorted(_proba_scores.items())]
        _table = tabulate([_inner_table],\
                          headers=['Probability','Amount Above','Score'],tablefmt="fancy_grid",floatfmt=".4f")
        return 'Results :\n%s\n'%_table
  
def FinalSeasonExperimentAllL(Experiment):
    '''
    A class that experiments the results of the classifier for last season (2015-2016).
    '''
    def __init__(self, dir_name, test=False):
        Experiment.__init__(self, dir_name, test=test)
        self.name = 'Final_Season'
    
    def load_params(self):
        best_param_exp = BestParamsExperiment("Best_Params", self._test)
        try:
            best_param_exp.load()
        except Exception as e:
            print 'Failed to load previous %s experiment\n. If you would like to run the %s experiment, Please type:\n Yes I am sure'
            ans = raw_input('>>>')
            if ans == 'Yes I am sure':
                best_param_exp.run()
            else:
                return
        self.estimators_params = {'DTC':best_param_exp._loaded_data['Tree'].best_params_,'RTC':best_param_exp._loaded_data['Forest'].best_params_}
    
        
    def run(self):
        Experiment.run(self)
        self.load_params()
        clf = DTC(**self.estimators_params['DTC'])
        clf = clf.fit(self.X,self.y)
        raw_curr_examples = []
        curr_tags = []
        for _league in LEAGUES:    
            self.cv.dbh.league = _league
            temp_examples, temp_tags = self.cv.dbh.create_examples(MAX_YEAR,lookback=15,True)
            raw_curr_examples += temp_examples
            curr_tags += temp_tags
        curr_examples = [_ex[0] for _ex in raw_curr_examples]
        result_tags = clf.predict(curr_examples)
        self._loaded_data = (clf.score(curr_examples, curr_tags),[],{i:0 for i in range(61)})
        amount_games_per_fix = {i:0 for i in range(61)}
        for i in range(len(raw_curr_examples)):
            _ex = raw_curr_examples[i]
            curr_fix = _ex[1]
            curr_result = _ex[2]
            self._loaded_data[1] += [(curr_result,result_tags[i])]
            amount_games_per_fix[curr_fix] += 1
            if result_tags[i] == curr_tags[i]:
                self._loaded_data[3][curr_fix] += 1
        
        for _k in self._loaded_data[3].keys():
            score = float(self._loaded_data[3][_k])
            self._loaded_data[3][_k] = score / amount_games_per_fix[_k]
        self.save(self._loaded_data)
        
    _begining_report = '''This experiment checks the best probability given by the Decision Tree from which  \
we start making the decisions.'''
            
    _ending_report = '''Done'''
    
    @property        
    def _no_detail(self):
        '''
        Reporting on low verbosity
        '''
        _proba_scores = {float(_k):self._loaded_data[_k] for _k in self._loaded_data.keys()}
        _inner_table = [[key,value[0],value[1]] for (key, value) in sorted(_proba_scores.items())]
        _table = tabulate([_inner_table],\
                          headers=['Probability','Amount Above','Score'],tablefmt="fancy_grid",floatfmt=".4f")
        return 'Results :\n%s\n'%_table
    
        
if __name__ == '__main__':
    args = ExperimentArgsParser().parse()
    _experiments = {'Best_Params':BestParamsExperiment,'AdaBoost':AdaBoostExperimet,'Best_Lookback':BestLookbackExperimet,'Best_Forest_Size':BestForestSizeExperiment,'Best_Proba':BestProbaForDecision,'Final_Year':FinalSeasonExperimentAllL,'Learning_Curve':LearningCurveExperiment,'Bayes':BayesExperiment}

    if args.action == 'run':
        _experiments[args.exp](dir_name=args.out_dir).run()
    else:
        _experiments[args.exp](dir_name=args.out_dir).report(verbosity=args.verbosity,outfile=args.outfile)
        
        