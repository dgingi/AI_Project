import sys
sys.path.append('..')
import unittest
from data.cross_validation import CrossValidation
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from utils.constants import LEAGUES
from experiments import BestParamsExperiment, AdaBoostExperimet, BestLookbackExperimet


class xTestCrossValidation(unittest.TestCase):


    def setUp(self):
        self.cv = CrossValidation(test=True)
        self.cv.load_data()


    def tearDown(self):
        pass


    def xtest_cv_generator(self):
        
        for train , test in self.cv._leagues_cross_validation():
            train , test
            clf = DTC()
            clf = clf.fit(train[0],train[1])
            clf.predict(test[0])

    def xtest_cv_scikit(self):
#         for train, test in self.cv.leagues_cross_validation:
#             train , test
        score = cross_val_score(DTC(), self.cv.complete_examples,self.cv.complete_tags,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        print 'Decision Tree',score , score.mean()

        
class TestExperiments(unittest.TestCase):
    def tearDown(self):
        unittest.TestCase.tearDown(self)
        from shutil import rmtree
        try:
            rmtree('Results/ut_results')
        except:
            pass
    
    def xtestBestParams(self):
        BestParamsExperiment('ut_results',True).run()
        
    def testAda(self):
        BestParamsExperiment('ut_results',True).run()
        AdaBoostExperimet('ut_results',True).run()
    
    def xtestLookback(self):
        BestLookbackExperimet('ut_results',True).run()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()