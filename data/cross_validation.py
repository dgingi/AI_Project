import sklearn

from dbhandler import DBHandler
import numpy
from utils.constants import LEAGUES, YEARS


class CrossValidation(object):
    
    def __init__(self,test=False,remote=True):
        self._test = test
        self.dbh = DBHandler(league=LEAGUES[0],test=self._test,remote=remote)
        self.data = {_l:{_y:(None,None) for _y in YEARS} for _l in LEAGUES}
        self._indces = {_l:0 for _l in LEAGUES}
    
    def load_data(self,lookback=2):
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
            train_data , test_data = self.create_indeces_leagues(train_leagues,test_league)
            res.append((numpy.array(train_data) , numpy.array(test_data)))
        self.cv_list = res
        
    def create_indeces_leagues(self,train,test):
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
        return self.cv_list
        
    
    def _leagues_cross_validation(self):
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
                 
            
        
    