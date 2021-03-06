import logging
import pickle
from progress.bar import ChargingBar
import pymongo
import sys
import numpy as np
from utils.constants import MAX_YEAR, MIN_YEAR, LEAGUES, MONTHS

sys.path.append('..')

class DBHandler(object):
    """
    This class is incharge of handling with all database realted functions.
    """
    def __init__(self,league,remote=False,test=False):
        """
        The Init function has the option to connect to the local database or our remote on-server database.
        """
        _host = '46.101.204.132' if remote else 'localhost'
        self.client = pymongo.MongoClient(host=_host)
        self._db = 'test' if test else 'leagues_db'
        self.DB = {temp_league:self.client[self._db][temp_league] for temp_league in LEAGUES}
        for col in self.DB:
            self.DB[col].create_index([('Year',pymongo.DESCENDING),('GName',pymongo.ASCENDING),('PName',pymongo.ASCENDING),('Fix',pymongo.DESCENDING)],\
                             unique = True)
        self.league = league
        self._test = test
        if remote:
            self.temp_DB = self.clone_db()
        else:
            self.temp_DB = self.DB 
    
    def clone_db(self):
        """
        This function clones the database from the server to the local computer inorder the save time in making queries.
        """
        from bson.son import SON
        temp_client = pymongo.MongoClient('localhost')
        _db = 'leagues_db' if not self._test else 'test'
        temp_DB = temp_client[_db]
        for league in LEAGUES:
            temp_col = temp_DB[league]
            if not self._test or temp_DB[league].find().count() == 0:
                temp_col.drop()
                temp_DB.command(SON([("cloneCollection","%s."%_db+league),("from",'46.101.204.132')]))
        return temp_DB
        
    
    def convert(self,data):
        """
        This function converts the data from our crawler to fit out database.  
        """
        return {name:{str(i):data[name][i] for i in range(1,2*len(data.keys())-1)} for name in data.keys()}
    
    def explode(self,data,year):
        """
        This function create all the table, making it by keys and saving as a list of entries.
        """
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
        """
        This function insert the data from the crawler to our database.
        
        This function uses both explde and convert methods.
        """
        try:
            self.DB[self.league].remove({'Year':int(year)})
        except Exception as e:
            pass
        try:
            self.DB[self.league].insert(self.explode(self.convert(data),year),continue_on_error=True)
        except Exception as e:
            logging.critical(str(e))
        

    def drop(self,year=None):
        """
        This function deletes data from our database.
        
        If year == None then it delets the entire data for self.league, otherwise just the requested year for self.league .
        """
        if year:
            self.DB[self.league].remove({'Year':int(year)})
        else:
            self.DB[self.league].drop()
    
       
    def create_examples(self,year,lookback=15,current=False):
        """
        This function creates all the examples for self.league, year.
        
        The examples are created using the given lookback.
        """
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
        
        from features.features import Features
        temp_DB = self.temp_DB
        all_teams_names = [g['_id'] for g in temp_DB[self.league].aggregate([{"$match":{"Year":int(year)}},{"$group":{"_id":"$GName"}}])]
        all_teams_dict = {name:{} for name in all_teams_names}
        features = Features(temp_DB[self.league],year,self.league)
        features_names = []
        prog_bar = ChargingBar('Creating examples for %s-%s'%(self.league,year),max=len(all_teams_dict))
        for team in all_teams_dict:
            res_by_all, res_by_non_avg = features.create_features(team,lookback)
            if not features_names: features_names = features.features_names
            update_all_teams_dict(res_by_all, all_teams_dict, team, True)
            update_all_teams_dict(res_by_non_avg, all_teams_dict, team, False)
            prog_bar.next()
        examples = []
        tags = []
        curr_examples = []
        prog_bar.finish()
        for team in all_teams_names:
            for fix in sorted(all_teams_dict[team]):
                if fix == 1 and all_teams_dict[team][fix]==[]:
                    continue
                curr_game = temp_DB[self.league].find_one({"GName":team,"Fix":fix,"Year":int(year)})
                if curr_game is None:
                    continue
                if curr_game["HA"]=="home":
                    vs_curr_game = temp_DB[self.league].find_one({"GName":curr_game["VS"],"VS":team,"HA":"away","Year":int(year)})
                    try:
                        vs_curr_fix = vs_curr_game["Fix"]
                    except TypeError as e:
                        vs_curr_fix = fix+1
                        all_teams_dict[curr_game["VS"]][vs_curr_fix] = []
                    if all_teams_dict[curr_game["VS"]][vs_curr_fix] == []:
                        continue
                    rel_all, rel_att, rel_def = relative_features(all_teams_dict[team][fix], all_teams_dict[curr_game["VS"]][vs_curr_fix], features_names)
                    examples += [np.array(all_teams_dict[team][fix])-np.array(all_teams_dict[curr_game["VS"]][vs_curr_fix])]
                    examples[-1] = np.concatenate((examples[-1],[rel_all, rel_att, rel_def]))
                    temp_dict = {"Ex":examples[-1],"Fix":curr_game["Fix"],"Res":curr_game["Result"],"Home":team,"Away":curr_game["VS"],"League":self.league}
                    curr_examples += [temp_dict]
                    tags += [curr_game["Tag"]]
        if not current:
            return examples,tags
        else:
            return curr_examples,tags
        

if __name__ == '__main__':
    for league in LEAGUES:
        for year in [str(i) for i in range(MIN_YEAR,MAX_YEAR+1)]:
            for month in MONTHS:
                try:
                    with open('../backup/%s-%s/%s.pckl'%(league,year,month),'r') as games:
                        data = pickle.load(games)
                        print 'Loading %s %s'%(league,year)
                        DBHandler(league).insert_to_db(data, year)
                except Exception,e:
                    continue