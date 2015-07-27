from time import clock
from pymongo import MongoClient
import numpy as np

class DBHandler():
    def __init__(self,league,year):
        self.client = MongoClient() #TODO: remote DB
        self.DB = self.client[league]
        self.col = self.DB[year]
        self.non_avg_keys = ["Position","PName","GName","Result","HA","_id","Tag","VS","Goals"]
        self.d_pos = ["GK","DR","DL","DC","DMC","DML","DMR","MR","MC","ML"]
        self.a_pos = ["FW","AR","AL","AC","AMC","AML","AMR"]
    
    def convert(self,data):
        """
        Convert the crawler data keys into string for insertion into MongoDB
        """
        return {name:{str(i):data[name][i] for i in range(1,2*len(data.keys())-1)} for name in data.keys()}
    
    def explode(self,data):
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
                        res.append(line)
        return res
    
    def insert_to_db(self,data):
        self.col.insert(self.explode(self.convert(data)))
        
    def drop(self,league,year=None):
        if year:
            self.client.DB.drop_collection(year)
        else:
            self.dlient.drop_database(league)
    
    def create_avg_aux(self,t_name,lookback=5):
        max_fix = max([g["Fix"] for g in self.col.find({"GName":t_name})])
        res_by_all = {i:self.create_avg_up_to_by_all(t_name, i, lookback) for i in range(1,max_fix+1)}
        res_by_fix = {i:self.create_avg_up_to_by_fix(t_name, i, lookback) for i in range(1,max_fix+1)}
        return res_by_all,res_by_fix
    
    """ Add avg. goals"""
    def create_avg_up_to_by_fix(self,t_name,fix,lookback=5):
        
        def all_features(res,by_f,group_q):
            temp_group_q = dict(group_q)
            for key in self.col.find_one().keys():
                if key in self.non_avg_keys:
                    continue
                index = "$"+key
                temp = ("avg_"+key+"_by_fix_"+by_f,{"$avg":index})
                res[temp[0]] = 0.0
                temp_group_q["$group"][temp[0]] = temp[1]
            return temp_group_q
            
        
        def create_pipe(t_name,fix,lookback,by_f,pos_list=[]):
            pipe = [{"$match":{"GName":t_name,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback}}}]
            if pos_list:
                pipe[0]["$match"]["Position"] = {"$in":pos_list}
            group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix"}}}
            pipe += [all_features(res,by_f, group_q)]
            return pipe
        
        def update_res(res,pipe):
            agg = self.col.aggregate(pipe)
            for cursor in agg:
                for key in cursor:
                    if key != "_id":
                        res[key]+=cursor[key]
                                
        res = {}
        curr_pipe = create_pipe(t_name,fix,lookback,"by_p") 
        update_res(res, curr_pipe)
        curr_pipe = create_pipe(t_name, fix, lookback, "by_pos", self.d_pos)
        update_res(res, curr_pipe)
        curr_pipe = create_pipe(t_name, fix, lookback, "by_pos", self.a_pos)
        update_res(res, curr_pipe)
        for key in res:
            if fix-lookback < 0:
                res[key] = res[key]/fix
            else:
                res[key] = res[key]/lookback
        return res  
    
    def create_avg_up_to_by_all(self,t_name,fix,lookback=5):
        pipe = [{"$match":{"GName":t_name,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback}}}]
        group_q = {"$group":{"_id":"$GName"}}
        for key in self.col.find_one().keys():
            if key in self.non_avg_keys:
                continue
            index = "$"+key
            temp = ("avg_"+key+"by_all",{"$avg":index})
            group_q["$group"][temp[0]] = temp[1]
        pipe += [group_q]
        for element in self.col.aggregate(pipe):
            del element["_id"]
            return element
    
    def create_examples(self):
        all_teams_names = [g['_id'] for g in self.col.aggregate([{"$group":{"_id":"$GName"}}])]
        all_teams_dict = {name:{} for name in all_teams_names}
        for team in all_teams_dict:
            res_by_all, res_by_fix = self.create_avg_aux(team)
            for fix in sorted(res_by_all):
                if fix == 1:
                    continue
                all_teams_dict[team][fix] = [res_by_all[fix][k] for k in sorted(res_by_all[fix])]
            for fix in sorted(res_by_fix):
                if fix == 1:
                    continue
                all_teams_dict[team][fix] += [res_by_fix[fix][k] for k in sorted(res_by_fix[fix])]
        examples = []
        tags = []
        for team in all_teams_names:
            for fix in sorted(all_teams_dict[team]):
                if fix == 1:
                    continue
                curr_game = self.col.find_one({"GName":team,"Fix":fix})
                if curr_game["HA"]=="home":
                    vs_curr_game = self.col.find_one({"GName":curr_game["VS"],"VS":team,"HA":"away"})
                    vs_curr_fix = vs_curr_game["Fix"]
                    examples += [np.array(all_teams_dict[team][fix])-np.array(all_teams_dict[curr_game["VS"]][vs_curr_fix])]
                    tags += [curr_game["Tag"]]
        return examples,tags
        
    def create_avg_per_game(self):
        pipe = [{"$match":{"Touches":{"$gt":0}}}]
        group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix"},"tot_player":{"$sum":1}}}
        for key in self.col.find_one().keys():
            index = "$"+key
            temp = ("avg_"+key,{"$avg":index})
            group_q["$group"][temp[0]] = temp[1]
        pipe += [group_q]
        return self.col.aggregate(pipe)
        
    
    
                

def PrintException():
    import linecache
    import sys
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)            

def timed(f):
    '''decorator for printing the timing of functions
    usage: 
    @timed
    def some_funcion(args...):'''
    
    def wrap(*x, **d):
        start = clock()
        res = f(*x, **d)
        print(f.__name__, ':', clock() - start)
        return res
    return wrap

    
    
