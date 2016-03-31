from dbhandler import DBHandler
import numpy
from utils.constants import LEAGUES, YEARS


class CrossValidation(object):
    """
    A class that implements our unique way to cross validate - 4 leagues for training, 1 for test -> 5 folds.
    
    Since our need to avoid a situation where we have tested a classifier on examples that are older from some of the examples used to fit the classifier,
    we've implemented our own Cross Validation that always learn from 4 leagues (all years defined) and test against the fifth league. 
    """
    
    def __init__(self,test=False,remote=False):
        """         
            Initialize a new CrossValidation instance.         
        
           :param test: if running in test mode or not
           :type test: boolean
           :param remote: whether to use the remote database that is configured in DBHandler
           :type remote: boolean
        """
        self._test = test
        self.dbh = DBHandler(league=LEAGUES[0],test=self._test,remote=remote)
        self.data = {_l:{_y:(None,None) for _y in YEARS} for _l in LEAGUES}
        self._indces = {_l:0 for _l in LEAGUES}
    
    def load_data(self,lookback=2):
        """
            Creates all the examples from the database based on the lookback parameter.
            
            Sets the complete_examples, complete_tags and cv_list attributes for later use.         
        
           :param lookback: how many previous games do we wish to include in the aggregation that creates the examples 
           :type lookback: integer
           :rtype: None
        """
        for league in LEAGUES:
            for year in YEARS:
                self.dbh.league = league
                self.data[league][year] = self.dbh.create_examples(year, lookback)
                self._indces[league] += len(self.data[league][year][0])
            else:
                if league != LEAGUES[0]:
                    self._indces[league] += self._indces[LEAGUES[LEAGUES.index(league)-1]]
        self.complete_examples = []
        self.complete_tags = []
        for _l in LEAGUES:
            for _y in YEARS:
                self.complete_examples.extend(self.data[_l][_y][0])
                self.complete_tags.extend(self.data[_l][_y][1])
        res = []
        for league in LEAGUES:
            train_leagues = list(set(LEAGUES) - set([league]))
            train_leagues.sort(key=LEAGUES.index)
            test_league = [league]
            train_data , test_data = self.create_indices_leagues(train_leagues,test_league)
            res.append((numpy.array(train_data) , numpy.array(test_data)))
        self.cv_list = res
    
    @property
    def _leagues_indeces(self):
        """
        Return a mapping of {league:examples and tags indices}
        
        :rtype: {league:range()}
        """
        pass
        d={}
        for league in LEAGUES:
            if league == LEAGUES[0]:
                d[league] = range(0,self._indces[league])
            else:
                d[league] = range(self._indces[LEAGUES[LEAGUES.index(league)-1]],self._indces[league])
        return d
    
    def create_indices_leagues(self,train,test):
        """
            Given a train set of examples and test set of examples, return a tuple of lists that holds the examples indices (for sklearn classes).
        
           :param train: training set of examples (with tags) (4 leagues) 
           :param test: testing set of examples (with tags) (1 league)
           :rtype: tuple(list,list)
        """
        _train , _test = [] , []
        for _l in train:
            if _l == LEAGUES[0]:
                _train.extend(range(0,self._indces[_l]))
            else:
                _train.extend(range(self._indces[LEAGUES[LEAGUES.index(_l)-1]],self._indces[_l]))
        for _t in test:
            if _t == LEAGUES[0]:
                _test.extend(range(0,self._indces[_t]))
            else:
                _test.extend(range(self._indces[LEAGUES[LEAGUES.index(_t)-1]],self._indces[_t]))
        return _train , _test
        
    @property
    def leagues_cross_validation(self):
        """
            Returns a list of 5 folds for usage as a cross validation instance from sklearn. 
        
           :rtype: list
        """
        return self.cv_list
        
    
    def _leagues_cross_validation(self):
        """
            Generator that yields tuples of ((train_examples,train_tags),(test_examples,test_tags)). 
        
           :rtype: tuple
        """
        for league in LEAGUES:
            train_leagues = list(set(LEAGUES) - set([league]))
            train_leagues.sort(key=LEAGUES.index)
            training_examples , training_tags = [] , []
            test_examples , test_tags = [] , []
            for _league in train_leagues:
                for year in YEARS:
                    training_examples.extend(self.data[_league][year][0])
                    training_tags.extend(self.data[_league][year][1])
            for year in YEARS:
                test_examples.extend(self.data[league][year][0])
                test_tags.extend(self.data[league][year][1])
            yield (training_examples,training_tags) , (test_examples,test_tags)
