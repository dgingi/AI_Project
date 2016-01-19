'''
Created on Jan 19, 2016

@author: dror
'''

import copy
from data.dbhandler import DBHandler
from features.features import Features
from utils.constants import MIN_YEAR,MAX_YEAR
            
class EXHandler():
    def __init__(self,league):
        self.league = league
        self.DBH = DBHandler(self.league)
        

    def get_features_names(self):
        D = DBHandler(self.league)
        team = [g['_id'] for g in D.cols["2012"].aggregate([{"$group":{"_id":"$GName"}}])][0]
        res_by_fix, res_by_non_avg = Features(D.cols,"2012").create_features(team)
        features_names = [k for k in sorted(res_by_fix[15])]
        features_names += [k for k in sorted(res_by_non_avg[15])]
        return features_names
    
    
    def get(self,year=None):
        examples = []
        tags = []

        for curr_year in range(MIN_YEAR,MAX_YEAR):
            if self.DBH.cols[str(curr_year)].count() != 0:
                temp_e,temp_t = self.DBH.create_examples(str(curr_year))
                examples += temp_e
                tags += temp_t

        return examples,tags
    
    def convert(self,ex,amount=3):
        new_ex = [copy.copy(x) for x in ex]
        for i in range(len(new_ex[0])):
            temp_list = [new_ex[j][i] for j in range(len(new_ex))]
            max_diff = max(temp_list)
            min_diff = min(temp_list)
            for j in range(len(new_ex)):
                res = new_ex[j][i]
                if res > 0:
                    for rel_amount in reversed(range(amount)):
                        if res >= (rel_amount * max_diff)/amount:
                            new_ex[j][i] = rel_amount + 1
                            break;
                elif res < 0:
                    for rel_amount in reversed(range(amount)):
                        if res <= (rel_amount * min_diff)/amount:
                            new_ex[j][i] = -1 * (rel_amount + 1)
                            break;
        return new_ex
    
    def predict(self,clf,examples,tags):
        res = clf.predict(examples)
        sum = 0
        for i in range(len(res)):
            if res[i]==tags[i]:
                sum+=1
        return (sum*1.0)/len(res)
    

