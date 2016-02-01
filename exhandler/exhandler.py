'''
Created on Jan 19, 2016

@author: dror
'''

import copy

from utils.constants import MIN_YEAR,MAX_YEAR
            
class EXHandler():
    def __init__(self,league,remote=True):
        from data.dbhandler import DBHandler
        self.league = league
        self.DBH = DBHandler(self.league,remote)
        

    def get_features_names(self):
        from features.features import Features
        team = [g['_id'] for g in self.DBH.DB[self.league].aggregate([{"$match":{"Year":2012}},{"$group":{"_id":"$GName"}}])][0]
        res_by_fix = Features(self.DBH.DB[self.league],2012,self.league).create_avg_up_to(team, 30, 15)
        res_by_non_avg = Features(self.DBH.DB[self.league],2012,self.league).create_avg_of_non_avg_f(team, 30, 15)
        features_names = [k for k in sorted(res_by_fix)]
        features_names += [k for k in sorted(res_by_non_avg)]
        features_names += ["relative_all_pos","relative_att_pos","relative_def_pos"]
        return features_names
    
    def split_to_train_and_test(self,ex,ta):
        from features.features import Features
        agg = self.DBH.DB[self.league].aggregate([{"$match":{"Year":MAX_YEAR-1}},{"$group":{"_id":{"GName":"$GName","VS":"$VS","Fix":"$Fix"}}}])
        idx = Features(self.DBH.DB,str(MAX_YEAR-1),self.league).get_agg_size(agg)
        idx /= 2
        X1,Y1 = ex[:-idx], ta[:-idx]
        X2,Y2 = ex[-idx:], ta[-idx:]
        return X1,Y1,X2,Y2
    
    def get(self,year=None,lookback=15):
        if not year:
            examples = []
            tags = []
            for curr_year in range(MIN_YEAR,MAX_YEAR):
                if self.DBH.DB[self.league].find_one({"Year":MIN_YEAR}):
                    temp_e,temp_t = self.DBH.create_examples(str(curr_year),lookback)
                    examples += temp_e
                    tags += temp_t
        else:
            examples, tags = self.DBH.create_examples(str(year))
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
    

