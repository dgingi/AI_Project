import unittest
from data.cross_validation import CrossValidation
from sklearn.tree import DecisionTreeClassifier as DTC



class TestCrossValidation(unittest.TestCase):


    def setUp(self):
        self.cv = CrossValidation(test=True)
        self.cv.load_data()


    def tearDown(self):
        pass


    def test_cv_generator(self):
        for train , test in self.cv.leagues_cross_validation():
            train , test
            clf = DTC()
            clf = clf.fit(train[0],train[1])
            clf.predict(test[0])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()