from datetime import datetime
from glob import glob
from os import  makedirs
import os
from os.path import exists, join as join_path
from pickle import dump , load
import pickle
from scipy.stats import ttest_rel
from sklearn.cross_validation import  cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.grid_search import RandomizedSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.multiclass import OneVsRestClassifier as OVRC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as DTC
from tabulate import tabulate
import warnings

from data.cross_validation import CrossValidation
from utils.argumet_parsers import ExperimentArgsParser
from utils.constants import LEAGUES, MAX_YEAR, MIN_YEAR
from utils.decorators import timed, move_to_root_dir


class Experiment():
    """
    Abstract Experiment class.
    
    Experiments classes provide basic functionality to running an experiment-
            allows saving and loading the results
            automatically loads data for cross validation
    """
    def __init__(self,dir_name,test=False):
        """
        Creates a new Experiment instance.
        
        Requires a dir_name (to save results) and a flag to indicate whether it's test environment or not.
        """
        self._dir_name = dir_name
        self.results_dir = join_path('Results',dir_name)
        self._test = test
        
        self._loaded_data = None
        
    def save(self,data):
        """
        Saves data into the results dir under the special format {experiment_name}.results
        """
        if not exists(self.results_dir):
            makedirs(self.results_dir)
        with open(join_path(self.results_dir,self.name+'.results'),'w') as _f:
            dump(data, _f)
        
    def load(self):
        """
        Loads results from previous runs into _loaded_data attribute.
        """
        _path = glob(join_path(self.results_dir,'%s.results'%self.name)).pop()
        with open(_path,'r') as _f:
            self._loaded_data = load(_f)
    
    def get_data(self):
        """
        Loads all the examples and tags needed for the experiment.
        
        Loads the all of the examples and tags, and also creates a cross validation for using in the estimators \ searches.
        """
        self.cv = CrossValidation(test=self._test)
        lookback = 2 if self._test else 15
        self.cv.load_data(lookback)
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags
    
    def run(self):
        """
        Runs the experiment.
        
        You should override this function in derived classes to configure the experiment
        """
        self.get_data()
        
    def load_params(self):
        """
        Loads the parameters of the experiment.
        
        You should override this in derived classes to configure the experiment parameters
        """
        raise NotImplementedError
      
    def t_test(self,original_measurements, measurements_after_alteration):
        """
        This is the paired T-test (for repeated measurements):
        Given two sets of measurements on the SAME data points (folds) before and after some change,
        Checks whether or not they come from the same distribution. 
        
        This T-test assumes the measurements come from normal distributions.
        
        Returns: 
            The probability the measurements come the same distributions
            A flag that indicates whether the result is statically significant
            A flag that indicates whether the new measurements are better than the old measurements
        """
        SIGNIFICANCE_THRESHOLD= 0.05
        
        test_value, probability= ttest_rel(original_measurements, measurements_after_alteration)
        is_significant= probability/2 < SIGNIFICANCE_THRESHOLD
        is_better= sum(original_measurements) < sum(measurements_after_alteration) #should actually compare averages, but there's no need since it's the same number of measurments.
        return probability/2 if is_better else 1-probability/2, is_significant, is_better
    
    def _load_prev_experiment(self,exp):
        """
        A function to load a previous experiment.
        """
        try:
            exp.load()
            return True
        except Exception as e:
            print 'Failed to load previous {ex} experiment\n. If you would like to run the {ex} experiment, Please type:\n Yes I am sure'.format(ex=exp.name)
            ans = raw_input('>>>')
            if ans == 'Yes I am sure':
                exp.run()
            else:
                return False
  
    def report(self,verbosity,outfile):
        """
        A method to report the experiment's results.
        """
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

class BestParamsExperiment(Experiment):
    """
    An experiment class to search for the best hyperparameters for an estimator.
    
    The search is done using RandomGridSearchCV, on the DecisionTreeClassifer estimator and the 
    RandomForestClassifier estimator. 
    """
    def __init__(self,dir_name,test=False):
        Experiment.__init__(self,dir_name,test)
        self.name = 'Best_Params'
        
    def load_params(self):
        """
        Loads the parameters for the experiment - the grids to search on.
        
        To change parameters or values for one of the estimators - change to correct field.
        """
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
    
    
    _begining_report = """This experiment performed a Randomized Search for 1000 iterations upon the hyper parameters grid for both the \
Decision Tree classifier and the Random Forest classifier."""
            
    _ending_report = """Done"""
    
    @property        
    def _no_detail(self):
        """
        Reporting on low verbosity - only tables
        """
        _def_exp = DefaultParamsExperiment('Default_Params')
        try:
            self._load_prev_experiment(_def_exp)
        except:
            pass
        tree_cross_scores = max(self._loaded_data['Tree'].grid_scores_,key= lambda x: x[1])[2]
        forest_cross_scores = max(self._loaded_data['Forest'].grid_scores_,key= lambda x: x[1])[2]
        def_tree_scores = _def_exp._loaded_data['Default_Tree']
        def_forest_scores = _def_exp._loaded_data['Default_Forest']
        
        def _evaluate(before,after):
            prob , is_sig , is_better = self.t_test(before[0], after[0])
            res = '\n'.join(['Paired T Test result between {before_name} and {after_name}: {proba:.5f}'.format(before_name = before[1],after_name=after[1],proba=prob),\
                             'The results {sig} statically significant, while {before_name} is {better} with score {before_score:.4f} than {after_name} with score {after_score:.4f}'.format(before_name = before[1],after_name=after[1],prob=prob,before_score=before[0].mean(),
                                                                after_score=after[0].mean(),sig = 'are' if is_sig else "aren't",
                                                                better= 'better' if not is_better else 'worse')])
            
            return res
        trees_t_test = _evaluate((def_tree_scores,'Decision Tree before search'), (tree_cross_scores,'Decision Tree after search'))
        forests_t_test = _evaluate((def_forest_scores,'Random Forest before search'), (forest_cross_scores,'Random Forest after search'))
        tree_forest_test = _evaluate((tree_cross_scores,'Decision Tree after search'), (forest_cross_scores,'Random Forest after search'))
        _res = '\n'.join(['Decision Tree before search accuracy score: {0:.4f}'.format(def_tree_scores.mean()),\
                          'Decision Tree after search accuracy score: {0:.4f}'.format(self._loaded_data['Tree'].best_score_),\
                          trees_t_test,'Best Decision Tree hyper parameters:\n'+str(self._loaded_data['Tree'].best_params_),
        'Random Forest before search accuracy score: {0:.4f}'.format(def_forest_scores.mean()),\
        forests_t_test,'Random Forest after search accuracy score: {0:.4f}'.format(self._loaded_data['Forest'].best_score_),\
        'Best Random Forest hyper parameters:\n'+str(self._loaded_data['Forest'].best_params_),tree_forest_test])
        
        return _res
    
    @property
    def _detail(self):
        """
        Plots the Decision Tree classifier after search and after fitting all of the data.
        
        Saves in results folder, in pdf format.
        """
        from sklearn.tree import export_graphviz
        from sklearn.externals.six import StringIO
        import pydot
        _names = ['avg_AccCrosses_by_all_pos_by_all_HA',\
                 'avg_AccCrosses_by_all_pos_by_home',\
                 'avg_AccLB_by_all_pos_by_all_HA',\
                 'avg_AccLB_by_all_pos_by_home',\
                 'avg_AccThB_by_all_pos_by_all_HA',\
                 'avg_AccThB_by_all_pos_by_home',\
                 'avg_AerialsWon_by_all_pos_by_all_HA',\
                 'avg_AerialsWon_by_all_pos_by_home',\
                 'avg_BlockedShots_by_def_pos_by_all_HA',\
                 'avg_BlockedShots_by_def_pos_by_home',\
                 'avg_Clearances_by_def_pos_by_all_HA',\
                 'avg_Clearances_by_def_pos_by_home',\
                 'avg_Crosses_by_all_pos_by_all_HA',\
                 'avg_Crosses_by_all_pos_by_home',\
                 'avg_Disp_by_att_pos_by_all_HA',\
                 'avg_Disp_by_att_pos_by_home',\
                 'avg_Dribbles_by_att_pos_by_all_HA',\
                 'avg_Dribbles_by_att_pos_by_home',\
                 'avg_Fouled_by_att_pos_by_all_HA',\
                 'avg_Fouled_by_att_pos_by_home',\
                 'avg_Fouls_by_def_pos_by_all_HA',\
                 'avg_Fouls_by_def_pos_by_home',\
                 'avg_Interceptions_by_def_pos_by_all_HA',\
                 'avg_Interceptions_by_def_pos_by_home',\
                 'avg_KeyPasses_by_att_pos_by_all_HA',\
                 'avg_KeyPasses_by_att_pos_by_home',\
                 'avg_LB_by_all_pos_by_all_HA',\
                 'avg_LB_by_all_pos_by_home',\
                 'avg_Offsides_by_att_pos_by_all_HA',\
                 'avg_Offsides_by_att_pos_by_home',\
                 'avg_PA%_by_all_pos_by_all_HA',\
                 'avg_PA%_by_all_pos_by_home',\
                 'avg_Passes_by_all_pos_by_all_HA',\
                 'avg_Passes_by_all_pos_by_home',\
                 'avg_ShotsOT_by_att_pos_by_all_HA',\
                 'avg_ShotsOT_by_att_pos_by_home',\
                 'avg_Shots_by_att_pos_by_all_HA',\
                 'avg_Shots_by_att_pos_by_home',\
                 'avg_ThB_by_all_pos_by_all_HA',\
                 'avg_ThB_by_all_pos_by_home',\
                 'avg_TotalTackles_by_def_pos_by_all_HA',\
                 'avg_TotalTackles_by_def_pos_by_home',\
                 'avg_Touches_by_all_pos_by_all_HA',\
                 'avg_Touches_by_all_pos_by_home',\
                 'avg_UnsTouches_by_att_pos_by_all_HA',\
                 'avg_UnsTouches_by_att_pos_by_home',\
                 'avg_Goals_by_fix_by_all_HA',\
                 'avg_Goals_by_fix_by_all_HA_specific',\
                 'avg_Goals_by_fix_by_home',\
                 'avg_Goals_by_fix_by_home_specific',\
                 'avg_Possession_rate_by_all_HA',\
                 'avg_Possession_rate_by_all_HA_specific',\
                 'avg_Possession_rate_by_home',\
                 'avg_Possession_rate_by_home_specific',\
                 'avg_Success_rate_by_all_HA',\
                 'avg_Success_rate_by_all_HA_specific',\
                 'avg_Success_rate_by_home',\
                 'avg_Success_rate_by_home_specific',\
                 'avg_received_Goals_by_fix_by_all_HA',\
                 'avg_received_Goals_by_fix_by_all_HA_specific',\
                 'avg_received_Goals_by_fix_by_home',\
                 'avg_received_Goals_by_fix_by_home_specific',\
                 'relative_all_pos',\
                 'relative_att_pos',\
                 'relative_def_pos']
        dot_data = StringIO()
        export_graphviz(self._loaded_data['Tree'].best_estimator_, out_file=dot_data,rounded=True,class_names=['Win (Home)','Draw','Win (Away)'],\
                        feature_names=_names)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf(join_path(self.results_dir,"best_tree.pdf"))
        return 'The plotted Decision Tree classifier is saved in {0}.'.format(os.path.abspath(join_path(self.results_dir,"best_tree.pdf")))

    def run(self):
        """
        Runs a RandomizedSearch on both DecisionTree and RandomForest classifiers.
        """
        Experiment.run(self)
        _grids = self.load_params()
        grid_tree = RandomizedSearchCV(DTC(), _grids['DTC'], n_jobs=-1, cv=self.cv.leagues_cross_validation,n_iter=1000)
        grid_tree.fit(self.cv.complete_examples,self.cv.complete_tags)
        grid_forest = RandomizedSearchCV(RFC(), _grids['RFC'], n_jobs=-1, cv=self.cv.leagues_cross_validation,n_iter=1000)
        grid_forest.fit(self.cv.complete_examples,self.cv.complete_tags)
        self._loaded_data = {'Tree':grid_tree,'Forest':grid_forest} 
        self.save(self._loaded_data)
        
class BayesExperiment(Experiment):
    """
    An experiment to test Naive Bayes classifier.
    """
    def __init__(self,dir_name,test=False):
        Experiment.__init__(self,dir_name,test)
        self.name = 'Bayes'
        
    def run(self):
        """
        Runs a cross validation on Naive Bayes Classfier.
        """
        Experiment.run(self)
        bayes_score = cross_val_score(GaussianNB(), self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        self._loaded_data = {'Bayes':bayes_score}
        self.save(self._loaded_data)
        
    _begining_report = """This experiment tried a Naive Bayes classifier."""
            
    _ending_report = """Done"""
    
    @property        
    def _no_detail(self):
        """
        Reporting on low verbosity - only results.
        """
        return '\n'.join(['Gaussian Naive Bayes accuracy score: {0:.4f}'.format(self._loaded_data['Bayes'].mean())])

class OneVsRestExperiment(Experiment):
    """
    An experiment to test Naive Bayes classifier.
    """
    def __init__(self,dir_name,test=False):
        Experiment.__init__(self,dir_name,test)
        self.name = 'OneVsRest'
    
    def load_params(self):
        """
        The load_params for this experiment loads the best params for the decision tree and random forest and also the best lookback.
        
        This params found by the experiment best_param, lookback by best_lookback.
        """
        best_param_exp = BestParamsExperiment("Best_Params", self._test)
        if not self._load_prev_experiment(best_param_exp): return False
        best_lookback_exp = BestLookbackExperimet("Best_Params", self._test)
        if not self._load_prev_experiment(best_lookback_exp): return False
        self.estimators_params = {'DTC':best_param_exp._loaded_data['Tree'].best_params_,'RFC':best_param_exp._loaded_data['Forest'].best_params_,\
                                  'Lookback':int(max([(best_lookback_exp._loaded_data[_lk][1].mean(),_lk) for _lk in best_lookback_exp._loaded_data])[1])}
        return True
    
    def get_data(self):
        """
        Loads all the examples and tags needed for the experiment - building the examples based on the lookback found in previous experiments.
        
        Loads the all of the examples and tags, and also creates cross validation for the classifiers..
        """
        self.cv = CrossValidation(test=self._test,remote=False)
        lookback = self.estimators_params['Lookback']
        self.cv.load_data(lookback)
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags
    
    @timed   
    def run(self):
        if not self.load_params(): 
            print 'Can not run- must load previous experiment'
            return
        Experiment.run(self)
        ovr_tree_score = cross_val_score(OVRC(DTC(**self.estimators_params['DTC']),-1),self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        ovr_forest_score = cross_val_score(OVRC(RFC(**self.estimators_params['RFC']),-1),self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        self._loaded_data = {'OvR Tree':ovr_tree_score,'OvR Forest':ovr_forest_score}
        self.save(self._loaded_data)
#         cross_size = 0
#         self._loaded_data = {'DTC':0,'RFC':0}
#         for train , test in self.cv._leagues_cross_validation():
#             cross_size += 1
#             
#             dt_estimator = OVRC(DTC(**self.estimators_params['DTC']),n_jobs=-1)
#             rf_estimator = OVRC(DTC(**self.estimators_params['RFC']),n_jobs=-1)
#             
#             dt_estimator = dt_estimator.fit(train[0], train[1])
#             rf_estimator = rf_estimator.fit(train[0], train[1])
#             
#             dt_score = dt_estimator.score(test[0], test[1])
#             rf_score = rf_estimator.score(test[0], test[1])
#             
#             self._loaded_data['DTC'] += dt_score
#             self._loaded_data['RFC'] += rf_score
#             
#         self._loaded_data['DTC'] = (self._loaded_data['DTC']*1.0) / cross_size
#         self._loaded_data['RFC'] = (self._loaded_data['RFC']*1.0) / cross_size
        
        
    _begining_report = """This experiment tried a OneVsRest classifier with both Decision Tree and Random Forest as base estimators."""
            
    _ending_report = """Done"""
    
    @property        
    def _no_detail(self):
        """
        Reporting on low verbosity - only results.
        """
        return '\n'.join(['One Vs Rest with Decision Tree accuracy score: {0:.4f}'.format(self._loaded_data['OvR Tree'].mean()),\
                          'One Vs Rest with Random Forest accuracy score: {0:.4f}'.format(self._loaded_data['OvR Forest'].mean())])
     
class DefaultParamsExperiment(Experiment):
    """
    An experiment to test Decision Tree and Random Forest classifiers without adjusting their respective hyper parameters.
    """
    def __init__(self,dir_name,test=False):
        Experiment.__init__(self,dir_name,test)
        self.name = 'Default_Params'
        
    def run(self):
        """
        Runs cross validation against Decision Tree and Random Forest classifiers. 
        """
        Experiment.run(self)
        tree_score = cross_val_score(DTC(), self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        forest_score = cross_val_score(RFC(), self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        self._loaded_data = {'Default_Tree':tree_score,'Default_Forest':forest_score}
        self.save(self._loaded_data)
        
    _begining_report = """This experiment tried both the Decision Tree and the Random Forest classifiers with default hyper parameters."""
            
    _ending_report = """Done"""
    
    @property        
    def _no_detail(self):
        """
        Reporting on low verbosity - only results.
        """
        return '\n'.join(['Decision Tree with default hyper parameters accuracy score: {0:.4f}'.format(self._loaded_data['Default_Tree'].mean()),\
                          'Random Forest with default hyper parameters accuracy score: {0:.4f}'.format(self._loaded_data['Default_Forest'].mean())])   
    
class LearningCurveExperiment(Experiment):
    """
    An experiment to test the learning curves of the classifiers (Decision Tree, Random Forest, Naive Bayes).
    
    The learning curve shows the classifier's accuracy correlation with the size of the training data.
    """
    def __init__(self,dir_name,test=False):
        Experiment.__init__(self,dir_name,test)
        self.name = 'Learning_Curve'
        
    def run(self):
        """
        Runs learning curve on all classifiers.
        """
        Experiment.run(self)
        best_param_exp = BestParamsExperiment(self._dir_name, self._test)
        self._load_prev_experiment(best_param_exp)
        tree_curve = learning_curve(DTC(**best_param_exp._loaded_data['Tree'].best_params_), self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        forest_curve = learning_curve(RFC(**best_param_exp._loaded_data['Forest'].best_params_), self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        bayes_curve = learning_curve(GaussianNB(), self.X, self.y,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        self._loaded_data = {'Tree_Curve':tree_curve,'Forest_Curve':forest_curve,'Bayes_Curve':bayes_curve}
        self.save(self._loaded_data)
        
    _begining_report = """This experiment checks the learning curve for all the classifiers. \n
Will plot the learning curves on screen."""
            
    _ending_report = """Done"""
    
    @property        
    def _no_detail(self):
        """
        Reporting on low verbosity- plots the learning curves
        """
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
        
class BestLookbackExperimet(Experiment):
    """
    An experiment to test what is thebest lookback for creating the examples.
    
    The lookback parameter defines how many previous games we are taking in consideration while generating the examples. 
    """
    def __init__(self, dir_name, test=False):
        Experiment.__init__(self, dir_name, test=test)
        self.name = 'Best_Lookback'
               
    def load_params(self,estimators=[]):
        """
        The load_params for this experiment loads the best params for the decision tree and random forest.
        
        This params found by the experiment best_param.
        """
        best_param_exp = BestParamsExperiment(self._dir_name, self._test)
        if not self._load_prev_experiment(best_param_exp): return False
        self.estimators = [DTC(**best_param_exp._loaded_data['Tree'].best_params_),\
                           RFC(**best_param_exp._loaded_data['Forest'].best_params_)]
        return True
        
    def get_data(self,lookback):
        self.cv = CrossValidation(test=self._test,remote=self._remote)
        self.cv.load_data(lookback)
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags
        
    def run(self):
        """
        For each fixture in range [1,66] (in jumps of 5), run a cross validation on both Decision Tree and Random Forest.
        """
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
            results[str(lookback)] = (dtc_score,rfc_score)
        self._loaded_data = results
        self.save(self._loaded_data)
        

    _begining_report = """This experiment checks both classifier's accuracy correlation with the lookback parameter \
for the creation on the examples."""
            
    _ending_report = """Done"""
    
    @property        
    def _no_detail(self):
        """
        Reporting on low verbosity - only tables
        """
        _dtc_scores = {int(_lk):self._loaded_data[_lk][0].mean() for _lk in self._loaded_data}
        _rfc_scores = {int(_lk):self._loaded_data[_lk][1].mean() for _lk in self._loaded_data}
        _table = tabulate([['Decision Tree']+[value for (key, value) in sorted(_dtc_scores.items())],\
                           ['Random Forest']+[value for (key, value) in sorted(_rfc_scores.items())]],\
                          headers=['Classifier / Lookback']+sorted(_dtc_scores),tablefmt="fancy_grid",floatfmt=".4f")
        return 'Cross validation scores for each classifier by lookback:\n%s\n'%_table
    
    @property
    def _detail(self):
        """
        Medium verbosity - show plotted graphs
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import numpy as np
            import matplotlib.pyplot as plt
    
    
            def plot_lookback_curve(title, ylim=None):
          
                plt.figure()
                plt.title(title)
                if ylim is not None:
                    plt.ylim(*ylim)
                plt.xlabel("Number of fixtures to look back")
                plt.ylabel("Score")
                _dtc_scores = {int(_lk):self._loaded_data[_lk][0] for _lk in self._loaded_data}
                _rfc_scores = {int(_lk):self._loaded_data[_lk][1] for _lk in self._loaded_data}
                
                train_sizes = sorted(_dtc_scores.keys())
                train_scores = [value for (_, value) in sorted(_dtc_scores.items())]
                test_scores = [value for (_, value) in sorted(_rfc_scores.items())] 
                
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
                         label="Decision Tree score")
                plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Random Forest score")
            
                plt.legend(loc="best")
                return plt
        
        plot_lookback_curve('Best Look Back')
        plt.show()
        return ''

class BestProbaForDecision(Experiment):
    """
    An experiment to test what is the best threshold from which we want to make the decision.
    """
    def __init__(self, dir_name, test=False):
        Experiment.__init__(self, dir_name, test=test)
        self.name = 'Best_Proba'
    
    def load_params(self):
        """
        The load_params for this experiment loads the best params for the decision tree and random forest and also the best lookback.
        
        This params found by the experiment best_param, lookback by best_lookback.
        """
        best_param_exp = BestParamsExperiment("Best_Params", self._test)
        if not self._load_prev_experiment(best_param_exp): return False
        best_lookback_exp = BestLookbackExperimet("Best_Params", self._test)
        if not self._load_prev_experiment(best_lookback_exp): return False
        self.estimators_params = {'DTC':best_param_exp._loaded_data['Tree'].best_params_,'RFC':best_param_exp._loaded_data['Forest'].best_params_,\
                                  'Lookback':int(max([(best_lookback_exp._loaded_data[_lk][1].mean(),_lk) for _lk in best_lookback_exp._loaded_data])[1])}
        return True
    
    def get_data(self):
        """
        Loads all the examples and tags needed for the experiment - building the examples based on the lookback found in previous experiments.
        
        Loads the all of the examples and tags, and also creates cross validation for the classifiers..
        """
        self.cv = CrossValidation(test=self._test)
        lookback = self.estimators_params['Lookback']
        self.cv.load_data(lookback)
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags

    
    def run(self):
        """
        For each probability p in [0.34,0.65] (in jumps of 0.01), make the decision only if classifier's probability is greater or equal to p.

        For each probability we calculate the amount of games that qulified and the score will be calculated by this amount.
        """
        if not self.load_params(): 
            print 'Can not run- must load previous experiment'
            return
        Experiment.run(self)
        if self._test:
            self.ranges = [0.34,0.35]
        else:
            self.ranges = [(float(_i)/100) for _i in range(34,66)]
        self._loaded_data = {}
        self._loaded_data['DTC'] = {k:(0,0) for k in self.ranges}
        self._loaded_data['RFC'] = {k:(0,0) for k in self.ranges}
        
        for _range in self.ranges:
            dt_decision_result = 0.0
            dt_score = 0
            dt_curr_decisions = 0
            
            rf_decision_result = 0.0
            rf_score = 0
            rf_curr_decisions = 0
            
            tot_games = 0
            
            for train , test in self.cv._leagues_cross_validation():  
                clf_dt = DTC(**self.estimators_params['DTC'])
                clf_dt = clf_dt.fit(train[0],train[1])
                
                clf_rf = RFC(**self.estimators_params['RFC'])
                clf_rf = clf_rf.fit(train[0],train[1])
                
                dt_res_tags = clf_dt.predict(test[0])
                dt_res_proba = clf_dt.predict_proba(test[0])
                
                rf_res_tags = clf_rf.predict(test[0])
                rf_res_proba = clf_rf.predict_proba(test[0])
                    
                for i in range(len(dt_res_tags)):
                    tot_games += 1
                    if max(dt_res_proba[i]) >= _range:
                        dt_curr_decisions += 1
                        if dt_res_tags[i] == test[1][i]:
                            dt_score += 1
                
                for i in range(len(rf_res_tags)):
                    if max(rf_res_proba[i]) >= _range:
                        rf_curr_decisions += 1
                        if rf_res_tags[i] == test[1][i]:
                            rf_score += 1 
                                   
            dt_decision_result = (dt_score*1.0)/dt_curr_decisions
            rf_decision_result = (rf_score*1.0)/rf_curr_decisions
            self._loaded_data['DTC'][_range] = (dt_curr_decisions,dt_decision_result)
            self._loaded_data['RFC'][_range] = (rf_curr_decisions,rf_decision_result)
            self._loaded_data["AG"] = tot_games
        self.save(self._loaded_data)
        
        
    _begining_report = """This experiment checks the best probability given by the Decision Tree from which  \
we start making the decisions."""
            
    _ending_report = """Done"""
    
    @property        
    def _no_detail(self):
        """
        Reporting on low verbosity - only tables
        """
        _proba_scores = {float(_k):(self._loaded_data['DTC'][_k],self._loaded_data['RFC'][_k]) for _k in self._loaded_data['DTC'].keys()}
        _inner_table = [[key,tup[0][0],tup[0][1],(float(tup[0][0])/self._loaded_data["AG"])*tup[0][1],tup[1][0],tup[1][1],(float(tup[1][0])/self._loaded_data["AG"])*tup[1][1]] for (key, tup) in sorted(_proba_scores.items())]
        _table = tabulate([data for data in _inner_table],\
                          headers=['Probability','#Examples Tree','Score Tree','AS DT','#Examples Forest','Score Forest','AS RF'],tablefmt="fancy_grid",floatfmt=".4f")
        return 'Results :\n%s\n'%_table
    
    @property
    def _detail(self):
        """
        Medium verbosity - show plotted graphs
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import matplotlib.pyplot as plt
    
    
            def plot_lookback_curve(title, ylim=None,_dtc_scores=[],_rfc_scores=[],kwargs={}):
          
                plt.figure()
                plt.title(title)
                if ylim is not None:
                    plt.ylim(*ylim)
                plt.xlabel(kwargs['xlabel'])
                plt.ylabel(kwargs['ylabel'])
                
                tree_train_sizes = sorted(_dtc_scores.keys())
                forest_train_sizes = sorted(_rfc_scores.keys())
                train_scores_mean = [value for (_, value) in sorted(_dtc_scores.items())] 
                test_scores_mean = [value for (_, value) in sorted(_rfc_scores.items())] 
                plt.grid()
            
                plt.plot(tree_train_sizes, train_scores_mean, 'o-', color="r",
                         label=kwargs['tree'])
                plt.plot(forest_train_sizes, test_scores_mean, 'o-', color="g",
                         label='forest')
            
                plt.legend(loc="best")
                return plt
        _dtc_scores = {float(_k):self._loaded_data['DTC'][_k][0] for _k in self._loaded_data['DTC']}
        _rfc_scores = {float(_k):self._loaded_data['RFC'][_k][0] for _k in self._loaded_data['RFC']}
        kwargs = {'xlabel':'Probability','ylabel':'#Examples who has a class with greater probability',\
                  'tree':'Decision Tree Classifier','forest':'Random Forest Classifer'}
        plot_lookback_curve("Number of examples by probability",None,_dtc_scores,_rfc_scores,kwargs)
        _dtc_scores = {self._loaded_data['DTC'][_k][0]:self._loaded_data['DTC'][_k][1] for _k in self._loaded_data['DTC']}
        _rfc_scores = {self._loaded_data['RFC'][_k][0]:self._loaded_data['RFC'][_k][1] for _k in self._loaded_data['RFC']}
        kwargs = {'xlabel':'Number of examples','ylabel':'Mean score',\
                  'tree':'Decision Tree Classifier','forest':'Random Forest Classifer'}
        plot_lookback_curve("Mean score by number of examples",None,_dtc_scores,_rfc_scores,kwargs)
        plt.show()
        return ''

class BestProbaDiffForDrawDecision(Experiment):
    """
    An experiment to test what is the best threshold from which we want to make the decision about a draw.
    """
    def __init__(self, dir_name, test=False):
        Experiment.__init__(self, dir_name, test=test)
        self.name = 'Best_Proba_Draw'
        
    def load_params(self):
        """
        The load_params for this experiment loads the best params for the decision tree and random forest and also the best lookback.
        
        This params found by the experiment best_param, lookback by best_lookback.
        """
        best_param_exp = BestParamsExperiment("Best_Params", self._test)
        if not self._load_prev_experiment(best_param_exp): return False
        best_lookback_exp = BestLookbackExperimet("Best_Params", self._test)
        if not self._load_prev_experiment(best_lookback_exp): return False
        best_proba_exp = BestProbaForDecision("Best_Proba", self._test)
        if not self._load_prev_experiment(best_proba_exp): return False
        self.estimators_params = {'DTC':best_param_exp._loaded_data['Tree'].best_params_,'RFC':best_param_exp._loaded_data['Forest'].best_params_,\
                                  'Lookback':int(max([(best_lookback_exp._loaded_data[_lk][1].mean(),_lk) for _lk in best_lookback_exp._loaded_data])[1]),\
                                  'Best_Proba':0.65}
        return True
    
    def get_data(self):
        """
        Loads all the examples and tags needed for the experiment - building the examples based on the lookback found in previous experiments.
        
        Loads the all of the examples and tags, and also creates cross validation for the classifiers..
        """
        self.cv = CrossValidation(test=self._test)
        lookback = self.estimators_params['Lookback']
        self.cv.load_data(lookback)
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags
    
    def run(self):
        """
        Using the predict_proba methods of the classifiers, we wish to test a new decision rule that will allow us to tag games as draw.
        
        The predict_proba methods return a list of distributions (of the possible tags) for all the examples in the test, e.g:
            [[0.1,0.2,0.7],[0.3,0.2,0.5],...]
        
        For each probability p in [0.01,0.19] (in jumps of 0.01), do a cross validation test with the following decision rule instead of the default one:
        
            if |P(Home Team Winning) - P(Away Team Winning)| <= p:
                tag as draw instead of choosing the tag with the highest probability
        """
        if not self.load_params(): 
            print 'Can not run- must load previous experiment'
            return
        Experiment.run(self)
        
        self.ranges = [(float(_i)/100) for _i in range(1,25)]
        self._loaded_data = {}
        self._loaded_data['DTC'] = {k:(0,0) for k in self.ranges}
        self._loaded_data['RFC'] = {k:(0,0) for k in self.ranges}
        
        for _range in self.ranges:
            dt_decision_result = 0.0
            dt_score = 0
            
            rf_decision_result = 0.0
            rf_score = 0

            for train , test in self.cv._leagues_cross_validation():  
                clf_dt = DTC(**self.estimators_params['DTC'])
                clf_dt = clf_dt.fit(train[0],train[1])
                
                clf_rf = RFC(**self.estimators_params['RFC'])
                clf_rf = clf_rf.fit(train[0],train[1])
                
                dt_res_tags = clf_dt.predict(test[0])
                dt_res_proba = clf_dt.predict_proba(test[0])
                
                rf_res_tags = clf_rf.predict(test[0])
                rf_res_proba = clf_rf.predict_proba(test[0])
                    
                for i in range(len(dt_res_tags)):
                    if max(dt_res_proba[i]) <= self.estimators_params['Best_Proba']: #if the max is greater than 0.65 - will trust the classifier
                        if abs(dt_res_proba[i][0]-dt_res_proba[i][2])<=_range: #diff between win - loss is very small - must be draw! 
                            dt_res_tags[i] = 0
                
                for i in range(len(rf_res_tags)):
                    if max(rf_res_proba[i]) <= self.estimators_params['Best_Proba']: #if the max is greater than 0.65 - will trust the classifier
                        if abs(rf_res_proba[i][0]-rf_res_proba[i][2])<=_range: #diff between win - loss is very small - must be draw! 
                            dt_res_tags[i] = 0 
                
                def _score(prediction,test):
                    '''
                    Given a prediction array and a test array, returns averaged score of prediction against test.
                    
                    Both arrays should be numpy arrays.
                    '''
                    from numpy import count_nonzero
                    return float(len(prediction)-count_nonzero(prediction-test))/len(prediction)
                
                dt_score += _score(dt_res_tags,test[1])
                rf_score += _score(rf_res_tags,test[1])
                                   
            dt_decision_result = (dt_score*1.0)/5
            rf_decision_result = (rf_score*1.0)/5
            self._loaded_data['DTC'][_range] = dt_decision_result
            self._loaded_data['RFC'][_range] = rf_decision_result
        self.save(self._loaded_data)
        
        
    _begining_report = """This experiment checks the best differnce between Win_proba and Lose_Proba such that for every lower diff  \
the decision will be a Draw."""
            
    _ending_report = """Done"""
    
    @property        
    def _no_detail(self):
        """
        Reporting on low verbosity - only tables
        """
        _proba_scores = {float(_k):(self._loaded_data['DTC'][_k],self._loaded_data['RFC'][_k]) for _k in self._loaded_data['DTC'].keys()}
        _inner_table = [[key,tup[0],tup[1]] for (key, tup) in sorted(_proba_scores.items())]
        _table = tabulate([data for data in _inner_table],\
                          headers=['Probability','Score DT','Score RF'],tablefmt="fancy_grid",floatfmt=".4f")
        return 'Results :\n%s\n'%_table
    
    @property
    def _detail(self):
        """
        Medium verbosity - show plotted graphs
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import numpy as np
            import matplotlib.pyplot as plt
    
    
            def plot_lookback_curve(title, ylim=None):
          
                plt.figure()
                plt.title(title)
                if ylim is not None:
                    plt.ylim(*ylim)
                plt.xlabel("Maximum difference between P(Home win) and P(Away win)")
                plt.ylabel("Mean Cross Validation Score")
                _dtc_scores = {float(_k):self._loaded_data['DTC'][_k] for _k in self._loaded_data['DTC']}
                _rfc_scores = {float(_k):self._loaded_data['RFC'][_k] for _k in self._loaded_data['RFC']}
                
                train_sizes = sorted(_dtc_scores.keys())
                train_scores_mean = [value for (_, value) in sorted(_dtc_scores.items())] 
                test_scores_mean = [value for (_, value) in sorted(_rfc_scores.items())] 
                plt.grid()
            
                plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Decision Tree score")
                plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Random Forest score")
            
                plt.legend(loc="best")
                return plt
        
        plot_lookback_curve('Correlation of difference and mean validation score')
        plt.show()
        return ''  
    
class FinalSeasonExperiment(Experiment):
    """
    An experiment to test the results of the classifier for last season (2015-2016).
    """
    def __init__(self, dir_name, test=False):
        Experiment.__init__(self, dir_name, test=test)
        self.name = dir_name
    
    def load_params(self):
        """
        The load_params for this experiment loads the best params for the decision tree and random forest and also the best lookback.
        
        This params found by the experiment best_param, lookback by best_lookback.
        """
        best_param_exp = BestParamsExperiment("Best_Params", self._test)
        if not self._load_prev_experiment(best_param_exp): return False
        best_lookback_exp = BestLookbackExperimet("Best_Params", self._test)
        if not self._load_prev_experiment(best_lookback_exp): return False
        self.estimators_params = {'DTC':best_param_exp._loaded_data['Tree'].best_params_,'RFC':best_param_exp._loaded_data['Forest'].best_params_,\
                                  'Lookback':int(max([(best_lookback_exp._loaded_data[_lk][1].mean(),_lk) for _lk in best_lookback_exp._loaded_data])[1])}
        return True
    
    def get_data(self):
        """
        Loads all the examples and tags needed for the experiment - building the examples based on the lookback found in previous experiments.
        
        Loads the all of the examples and tags, and also creates cross validation for the classifiers..
        """
        self.cv = CrossValidation(test=self._test)
        lookback = self.estimators_params['Lookback']
        self.cv.load_data(lookback)
        self.X = self.cv.complete_examples
        self.y = self.cv.complete_tags
    
    def run(self):
        """
        Runs prediction against the current season (2015-2016).
        
        Tries both building a classifier from all the leagues and predict, and building a specific classifier for each league and predict only that league.
        """
        if not self.load_params():
            print 'Can not run- must load previous experiment'
            return
        if self.name == "Final_Season":
            Experiment.run(self)
            clf = RFC(**self.estimators_params['RFC'])
            clf = clf.fit(self.X,self.y)
        else:
            self.cv = CrossValidation(test=self._test)
            clf = RFC(**self.estimators_params['RFC'])
        
        self._loaded_data = {}   
        
        for _league in LEAGUES:
            self.cv.dbh.league = _league

            if self.name == "Final_Season_S":
                self.X = []
                self.y = []
                for year in range(MIN_YEAR,MAX_YEAR):
                    temp_ex, temp_ta = self.cv.dbh.create_examples(year,lookback=self.estimators_params['Lookback'],current=False)
                    self.X += temp_ex
                    self.y += temp_ta
                clf = clf.fit(self.X,self.y)
            
            raw_curr_examples = []
            curr_tags = []    
            raw_curr_examples, curr_tags = self.cv.dbh.create_examples(MAX_YEAR,lookback=self.estimators_params['Lookback'],current=True)
            
            curr_examples = [_ex["Ex"] for _ex in raw_curr_examples]
            result_tags = clf.predict(curr_examples)
            result_proba = clf.predict_proba(curr_examples)
            self._loaded_data[_league] = {"score":clf.score(curr_examples, curr_tags),"array":[],"dict":{i:0 for i in range(61)}}
            amount_games_per_fix = {i:0 for i in range(61)}
            for i in range(len(raw_curr_examples)):
                _ex = raw_curr_examples[i]
                curr_fix = _ex["Fix"]
                curr_result = _ex["Res"]
                self._loaded_data[_league]["array"] += [(_ex["League"],_ex["Home"],_ex["Away"],curr_result,result_tags[i],result_proba[i])]
                amount_games_per_fix[curr_fix] += 1
                if result_tags[i] == curr_tags[i]:
                    self._loaded_data[_league]["dict"][curr_fix] += 1
            
            for _k in self._loaded_data[_league]["dict"].keys():
                score = float(self._loaded_data[_league]["dict"][_k])
                if amount_games_per_fix[_k] == 0:
                    self._loaded_data[_league]["dict"][_k] = score
                else:
                    self._loaded_data[_league]["dict"][_k] = score / amount_games_per_fix[_k]
        self.save(self._loaded_data)
    
    
    try:
        with open('last_crawl.date','r') as _date_file:
            _last_crawl = pickle.load(_date_file)
    except IOError:
        _last_crawl = datetime(2016,3,12)
    _begining_report = "This experiment checks the results of the classifier for last season in 2 ways: One classifier from all the leagues and one specific classifier for each league.\nLast games are from {}".format(_last_crawl.strftime("%A %d. %B %Y"))
            
    _ending_report = """Done"""
    
    @property        
    def _no_detail(self):
        """
        Reporting on low verbosity - generates prediction in for each league (all games in current season), and prints averaged accuracy for each fixture.
        
        For each fixture we get averaged accuracy for all leagues and average accuracy per league.
        """
        all_scores = {}
        all_scores_league = {}
        amount_overlap = {}
        flag_first = True
        for league in LEAGUES:
            _scores = {int(_lk):self._loaded_data[league]["dict"][_lk] for _lk in self._loaded_data[league]["dict"]}
            _scores[0] = self._loaded_data[league]["score"]
            all_scores_league[league] = _scores
            if flag_first:
                for _k in _scores:
                    if _scores[_k] == 0.0:
                        continue
                    amount_overlap[_k] = 1
                    all_scores[_k] = _scores[_k]
                flag_first = False
            else:
                for _k in _scores:
                    if _scores[_k] == 0.0:
                        continue
                    if _k not in all_scores.keys():
                        amount_overlap[_k] = 1
                        all_scores[_k] = _scores[_k]
                    else:
                        amount_overlap[_k] += 1
                        all_scores[_k] += _scores[_k]
        for _k in all_scores:
            all_scores[_k] = float(all_scores[_k]) / amount_overlap[_k]
        _inner_table = [[key,all_scores[key]] for key in sorted(all_scores.keys())]
        for i in range(len(_inner_table)):
            _inner_table[i] += [all_scores_league[_l][i] for _l in sorted(LEAGUES)]
        for i in range(1,len(_inner_table)):
            for j in range(len(_inner_table[i])):
                if _inner_table[i][j] == 0.0:
                    _inner_table[i][j] = 'Not Played'
        curr_headers = ['Fix','Score'] + [_l for _l in sorted(LEAGUES)]
        _table = tabulate([data for data in _inner_table],\
                            headers=curr_headers,tablefmt="fancy_grid",floatfmt=".4f")
        from numpy import array
        print 'Results for experiment %s:\n%s\nMean accuracy for all leagues: %.4f'%(self.name,_table,array([data[1] for data in _inner_table]).mean())
        return ''
        
    @property
    def _detail(self):
        """
        Medium verbosity - save external files 
        
        For each league we save {fix,success_rate} , {team_a,team_b,result,tag,proba(-1),proba(0),proba(1)
        """
        all_scores = {}
        amount_overlap = {}
        flag_first = True
        for league in LEAGUES:
            _scores = {int(_lk):self._loaded_data[league]["dict"][_lk] for _lk in self._loaded_data[league]["dict"]}
            _scores[0] = self._loaded_data[league]["score"]
            if flag_first:
                for _k in _scores:
                    if _scores[_k] == 0.0:
                        continue
                    amount_overlap[_k] = 1
                    all_scores[_k] = _scores[_k]
                flag_first = False
            else:
                for _k in _scores:
                    if _scores[_k] == 0.0:
                        continue
                    if _k not in all_scores.keys():
                        amount_overlap[_k] = 1
                        all_scores[_k] = _scores[_k]
                    else:
                        amount_overlap[_k] += 1
                        all_scores[_k] += _scores[_k]
            _inner_table = [[key,_scores[key]] for key in sorted(_scores.keys()) if _scores[key] != 0.0]
            _table = tabulate([data for data in _inner_table],\
                              headers=['Fix','Score'],tablefmt="fancy_grid",floatfmt=".4f")
            with open(os.path.join(self.results_dir,league+"_fix_res.txt"),'w') as output:
                output.write(_table.encode("utf-8"))
                output.close()
            
            _inner_table = [[elem[1],elem[2],elem[3][0]+"-"+elem[3][1],elem[4],elem[5][0],elem[5][1],elem[5][2]] for elem in self._loaded_data[league]["array"] if elem[0]==league]
            _table = tabulate([data for data in _inner_table],\
                          headers=['Home','Away','Result','Tag','-1','0','1'],tablefmt="fancy_grid")
            with open(os.path.join(self.results_dir,league+"_results.txt"),'w') as output:
                output.write(_table.encode("utf-8"))
                output.close()

class FinalSeasonAux(Experiment):
    """
    Auxiliary experiment around the Final Season experiment modes.
    """
    def __init__(self, dir_name, test=False):
        Experiment.__init__(self, dir_name, test=test)
        self.name = dir_name
        
    @timed
    def run(self):
        FinalSeasonExperiment("Final_Season",self._test).run()
        FinalSeasonExperiment("Final_Season_S",self._test).run()
        
    def report(self, verbosity, outfile):
        FinalSeasonExperiment("Final_Season",self._test).report(verbosity, outfile)
        FinalSeasonExperiment("Final_Season_S",self._test).report(verbosity, outfile)
        
         
if __name__ == '__main__':
    with move_to_root_dir():
        args = ExperimentArgsParser().parse()
        _experiments = {'Best_Params':BestParamsExperiment,'Best_Lookback':BestLookbackExperimet,'Best_Proba':BestProbaForDecision,\
                    'Best_Proba_Diff':BestProbaDiffForDrawDecision,'Final_Season':FinalSeasonAux,'OVR':OneVsRestExperiment,\
                    'Learning_Curve':LearningCurveExperiment,'Bayes':BayesExperiment,'Default_Params':DefaultParamsExperiment}
    
        if args.action == 'run':
            _experiments[args.exp](dir_name=args.out_dir).run()
        else:
            _experiments[args.exp](dir_name=args.out_dir).report(verbosity=args.verbosity,outfile=args.outfile)
       
        