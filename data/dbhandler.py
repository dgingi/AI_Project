'''
A module to handle the backend of the project.

@author: Dror Porat & Ory Jonay
'''
import sys
sys.path.append('..')
import pickle
from pymongo import MongoClient

import numpy as np
from old_utils import Features, EXHandler
from utils.constants import MAX_YEAR, MIN_YEAR, LEAGUES
from utils.decorators import timed


class DBHandler():
    def __init__(self,league,remote=True):
        _host = '46.101.204.132' if remote else 'localhost'
        self.client = MongoClient(host=_host)
        self.DB = self.client[league]
        self.cols = {str(year):self.DB[str(year)] for year in range(MIN_YEAR,MAX_YEAR)}
        self.cols["all"] = self.DB["all"]
        self.league = league
    
    def convert(self,data):
        return {name:{str(i):data[name][i] for i in range(1,2*len(data.keys())-1)} for name in data.keys()}
    
    def explode(self,data,year):
        res = []
        for team in data:
            for fix in sorted(data[team]):
                if data[team][fix] == {}:
                    continue
                for player in data[team][fix]['Players']:
                    line = {}
                    for table in data[team][fix]['Players'][player]:
                        for key in data[team][fix]['Players'][player][table]:
                            line[key] = data[team][fix]['Players'][player][table][key]
                    else:
                        line['PName'] = player
                        line['GName'] = team
                        line['Fix'] = int(fix)
                        line['HA'] = data[team][fix]['HA']
                        line['Result'] = data[team][fix]['Result']
                        line['Possession'] = float(data[team][fix]['Possession'].split('%')[0])
                        line['VS'] = data[team][fix]['VS']
                        line['Tag'] = int(data[team][fix]['Tag'])
                        line['Year'] = int(year)
                        res.append(line)
        return res
    
    def insert_to_db(self,data,year):
        if self.cols[year].count() != 0:
            self.drop(year)
        res = self.explode(self.convert(data),year)
        self.cols[year].insert(res)
        self.cols["all"].insert(res)
        
    def drop(self,year=None):
        if year:
            self.client[self.league].drop_collection(year)
        else:
            self.client.drop_database(self.league)
    
    @timed        
    def create_examples(self,year,lookback=15):
        
        def update_all_teams_dict(res,all_teams_dict,team,first):
            for fix in sorted(res):
                if fix == 1 and res[fix] == {}:
                    all_teams_dict[team][fix] = []
                    continue
                if first:
                    all_teams_dict[team][fix] = [res[fix][k] for k in sorted(res[fix])]
                else:
                    all_teams_dict[team][fix] += [res[fix][k] for k in sorted(res[fix])]
        
        def relative_features(arr1,arr2,fn):
            combined_list_all_1 = [value for (value,key) in zip(arr1,fn) if key.split("all_pos")>1 ]
            combined_list_att_1 = [value for (value,key) in zip(arr1,fn) if key.split("att_pos")>1 ]
            combined_list_def_1 = [value for (value,key) in zip(arr1,fn) if key.split("def_pos")>1 ]
            
            combined_list_all_2 = [value for (value,key) in zip(arr2,fn) if key.split("all_pos")>1 ]
            combined_list_att_2 = [value for (value,key) in zip(arr2,fn) if key.split("att_pos")>1 ]
            combined_list_def_2 = [value for (value,key) in zip(arr2,fn) if key.split("def_pos")>1 ]
            
            all_rel = [1 for (val1,val2) in zip (combined_list_all_1,combined_list_all_2) if val1 > val2]
            att_rel = [1 for (val1,val2) in zip (combined_list_att_1,combined_list_att_2) if val1 > val2]
            def_rel = [1 for (val1,val2) in zip (combined_list_def_1,combined_list_def_2) if val1 > val2]
            
            return float(len(all_rel))/len(combined_list_all_1), float(len(att_rel))/len(combined_list_att_1), float(len(def_rel))/len(combined_list_def_1)
            
            
        all_teams_names = [g['_id'] for g in self.cols[year].aggregate([{"$group":{"_id":"$GName"}}])]
        all_teams_dict = {name:{} for name in all_teams_names}
        features = Features(self.cols,year)
        features_names = EXHandler(self.league).get_features_names()
        for team in all_teams_dict:
            res_by_all, res_by_non_avg = features.create_features(team,lookback)
            update_all_teams_dict(res_by_all, all_teams_dict, team, True)
            update_all_teams_dict(res_by_non_avg, all_teams_dict, team, False)
        examples = []
        tags = []
        for team in all_teams_names:
            for fix in sorted(all_teams_dict[team]):
                if fix == 1 and all_teams_dict[team][fix]==[]:
                    continue
                curr_game = self.cols[year].find_one({"GName":team,"Fix":fix})
                if curr_game["HA"]=="home":
                    vs_curr_game = self.cols[year].find_one({"GName":curr_game["VS"],"VS":team,"HA":"away"})
                    vs_curr_fix = vs_curr_game["Fix"]
                    if all_teams_dict[curr_game["VS"]][vs_curr_fix] == []:
                        continue
                    rel_all, rel_att, rel_def = relative_features(all_teams_dict[team][fix], all_teams_dict[curr_game["VS"]][vs_curr_fix], features_names)
                    examples += [np.array(all_teams_dict[team][fix])-np.array(all_teams_dict[curr_game["VS"]][vs_curr_fix])]
                    examples[-1] += [rel_all, rel_att, rel_def]
                    tags += [curr_game["Tag"]]
        return examples,tags
        



if __name__ == '__main__':
    for league in LEAGUES:
        for year in [str(i) for i in range(MIN_YEAR,MAX_YEAR)]:
            with open('../backup/%s-%s/May.pckl'%(league,year),'rb') as games:
                data = pickle.load(games)
                print 'Loading %s %s'%(league,year)
                DBHandler(league).insert_to_db(data, year)