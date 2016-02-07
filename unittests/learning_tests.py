import sys
sys.path.append('..')
import unittest
from data.cross_validation import CrossValidation
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.cross_validation import cross_val_score
from utils.constants import LEAGUES


class TestCrossValidation(unittest.TestCase):


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

    def test_cv_scikit(self):
#         for train, test in self.cv.leagues_cross_validation:
#             train , test
        score = cross_val_score(DTC(), self.cv.complete_examples,self.cv.complete_tags,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        print 'Decision Tree',score , score.mean()
        score = cross_val_score(RFC(n_estimators=300), self.cv.complete_examples,self.cv.complete_tags,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        print 'Random Forest 300',score , score.mean()
        score = cross_val_score(ABC(n_estimators=100), self.cv.complete_examples,self.cv.complete_tags,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
        print 'AdaBoost',score , score.mean()
        
        for i in range(50,1001,50):
            score = cross_val_score(RFC(n_estimators=i), self.cv.complete_examples,self.cv.complete_tags,  cv=self.cv.leagues_cross_validation,n_jobs=-1)
            print 'Random Forest %d'%i,score , score.mean()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()