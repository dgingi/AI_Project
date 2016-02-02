import unittest
from data.cross_validation import CrossValidation
from sklearn.tree import DecisionTreeClassifier as DTC
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
        score = cross_val_score(DTC(), self.cv.complete_examples,self.cv.complete_tags,  cv=self.cv.leagues_cross_validation)
        print score , sum(score)/len(score)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()