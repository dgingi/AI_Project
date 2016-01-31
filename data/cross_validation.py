import sklearn

from dbhandler import DBHandler
from utils.constants import LEAGUES, YEARS


class CrossValidation(object):
    
    def __init__(self,test=False):
        self._test = test
        self.dbh = DBHandler(league=LEAGUES[0],test=self._test)
        self.data = {_l:{_y:(None,None) for _y in YEARS} for _l in LEAGUES}
    
    def load_data(self,lookback=2):
        for league in LEAGUES:
            for year in YEARS:
                self.dbh.league = league
                self.data[league][year] = self.dbh.create_examples(year, lookback)
    
    def leagues_cross_validation(self):
        for league in LEAGUES:
            train_leagues = set(LEAGUES) - set([league])
            test_league = [league]
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
                
            
        
    