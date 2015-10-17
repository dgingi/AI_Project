from time import clock
from pymongo import MongoClient
import numpy as np
from boto.sdb.db.sequence import double

class Features():
    def __init__(self,data):
        self.col = data.col
        self.non_avg_keys = ["Position","PName","GName","Result","HA","_id","Tag","VS","Goals","Fix"]
        self.d_pos = ["GK","DR","DL","DC","DMC","DML","DMR","MR","MC","ML"]
        self.a_pos = ["FW","AR","AL","AC","AMC","AML","AMR"]
        self.o_pos = ["Sub"]

    def create_features(self,t_name,lookback=5):
        max_fix = max([g["Fix"] for g in self.col.find({"GName":t_name})])
        res_by_all = {i:self.create_avg_up_to("by_all_fix",t_name, i, lookback) for i in range(1,max_fix+1)}
        res_by_fix = {i:self.create_avg_up_to("by_fix",t_name, i, lookback) for i in range(1,max_fix+1)}
        res_by_non_avg = {i:self.create_avg_of_non_avg_f(t_name, i, lookback) for i in range(1,max_fix+1)}
        return res_by_all,res_by_fix,res_by_non_avg
           
    def get_curr_HA(self,t_name,fix):
        return self.col.find_one({"GName":t_name,"Fix":fix})["HA"]
    
    def get_agg_size(self,agg):
                agg_size = 0
                for cursor in agg:
                    agg_size += 1
                return 1 if agg_size == 0 else agg_size
        
    def create_avg_of_non_avg_f(self,t_name,fix,lookback=5):
        
        def update_avg_goals_scored(res,t_name,fix,by_loc,HA_list,lookback=5):
            if fix==1:
                #TODO same as beneath
                return
            pipe = [{"$match":{"GName":t_name,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback},"HA":{"$in":HA_list}}}]
            group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix"},"avg_Goals_by_fix"+by_loc:{"$sum":"$Goals"}}}
            pipe += [group_q]
            res["avg_Goals_by_fix"+by_loc] = 0.0
            agg = self.col.aggregate(pipe)
            num_of_games = self.get_agg_size(agg)
            agg = self.col.aggregate(pipe) 
            for cursor in agg:
                for key in cursor:
                    if key!="_id":
                        res["avg_Goals_by_fix"+by_loc] += cursor[key]
            res["avg_Goals_by_fix"+by_loc] /= num_of_games
        
        def update_avg_received_goals(res,t_name,fix,by_loc,HA_list,lookback=5):
            
            def select_recieved_goals(HA,result):
                if HA=="home":
                    return int(result[1])
                else:
                    return int(result[0])
            
            if fix==1:
                #TODO same as beneath
                return
                
            pipe = [{"$match":{"GName":t_name,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback},"HA":{"$in":HA_list}}}]
            group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix","HA":"$HA","Result":"$Result"}}}
            pipe += [group_q]
            res["avg_received_Goals_by_fix"+by_loc] = 0.0
            agg = self.col.aggregate(pipe)
            num_of_games = self.get_agg_size(agg)
            agg = self.col.aggregate(pipe)
            for cursor in agg:
                for key in cursor:
                    res["avg_received_Goals_by_fix"+by_loc] += select_recieved_goals(cursor[key]["HA"], cursor[key]["Result"])
            res["avg_received_Goals_by_fix"+by_loc] /= num_of_games
        
        def update_avg_success_rate(res,t_name,fix,by_loc,HA_list,lookback=5):
            
            pipe = [{"$match":{"GName":t_name,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback},"HA":{"$in":HA_list}}}]
            group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix","HA":"$HA","Result":"$Result"}}}
            pipe += [group_q]
            res["avg_Success_rate"+by_loc] = 0.0
            agg = self.col.aggregate(pipe)
            num_of_games = self.get_agg_size(agg)
            agg = self.col.aggregate(pipe)
            for cursor in agg:
                res["avg_Success_rate"+by_loc] += select_recieved_goals(cursor[key]["HA"], cursor[key]["Result"])
            res["avg_Success_rate"+by_loc] /= num_of_games
            
        res = {}
        curr_HA = self.get_curr_HA(t_name, fix) 
        
        update_avg_goals_scored(res,t_name, fix, "_by_all_HA", ["home","away"], lookback)
        update_avg_goals_scored(res,t_name, fix, "_by_"+curr_HA, [curr_HA], lookback)
        update_avg_received_goals(res, t_name, fix, "_by_all_HA", ["home","away"], lookback)
        update_avg_received_goals(res, t_name, fix, "_by_"+curr_HA, [curr_HA], lookback)
        
        return res
        
    def create_avg_up_to(self,by_avg,t_name,fix,lookback=5):
        
        def decide_pos_list(str):
            pos = str.split('_')[1]
            if pos == 'all':
                return []
            elif pos == 'def':
                return self.d_pos
            elif pos == 'att':
                return self.a_pos
            else:
                return self.o_pos
        
        def make_pos_list(str):
            if str == "by_all_fix":
                return ["by_all_pos","by_sub_pos"]
            else:
                return ["by_all_pos","by_def_pos","by_att_pos","by_sub_pos"]
           
        def update_res(res,t_name,fix,lookback,by_avg,by_pos,by_loc,HA_list,pos_list):
            
            def create_pipe(res,t_name,fix,lookback,by_avg,by_pos,by_loc,HA_list,pos_list):
                
                def all_features(res,by_avg,by_pos,by_loc,group_q):
                    temp_group_q = dict(group_q)
                    for key in self.col.find_one().keys():
                        if key in self.non_avg_keys:
                            continue
                        index = "$"+key
                        temp = ("avg_"+key+"_"+by_avg+"_"+by_pos+"_"+by_loc,{"$avg":index})
                        res[temp[0]] = 0.0
                        temp_group_q["$group"][temp[0]] = temp[1]
                    return temp_group_q
                
                pipe = [{"$match":{"GName":t_name,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback},"HA":{"$in":HA_list}}}]
                if pos_list:
                    pipe[0]["$match"]["Position"] = {"$in":pos_list}
                group_q = {}
                if by_avg == "by_all_fix":
                    group_q = {"$group":{"_id":{"GName":"$GName"}}}
                else:
                    group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix"}}}
                pipe += [all_features(res,by_avg,by_pos,by_loc, group_q)]
                return pipe
            
            if fix==1:
                #TODO - check for history if availabe and act on it + to add if fox-lookback<0 then add by history
                return
            
            temp_res = {}
            pipe = create_pipe(temp_res, t_name, fix, lookback,by_avg, by_pos, by_loc, HA_list, pos_list)
            agg = self.col.aggregate(pipe)
            num_of_games = self.get_agg_size(agg)
            agg = self.col.aggregate(pipe)
            for cursor in agg:
                for key in cursor:
                    if key != "_id":
                        temp_res[key]+=cursor[key]
            if by_avg == "by_fix":
                for key in temp_res:
                    temp_res[key] /= num_of_games
            res.update(temp_res)
                                
        res = {}
        curr_HA = self.get_curr_HA(t_name, fix)
        pos_list = make_pos_list(by_avg)
         
        for pos in pos_list:
            curr_pos_list = decide_pos_list(pos)
            update_res(res,t_name,fix,lookback,by_avg,pos, "by_all_HA",["home","away"],curr_pos_list)
            update_res(res,t_name,fix,lookback,by_avg,pos, "by_"+curr_HA, [curr_HA],curr_pos_list)
             
        return res                

        
class DBHandler():
    def __init__(self,league,year):
        self.client = MongoClient() #TODO: remote DB
        self.DB = self.client[league]
        self.col = self.DB[year]
    
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
            self.client.drop_database(league)
    
    def create_examples(self):
        all_teams_names = [g['_id'] for g in self.col.aggregate([{"$group":{"_id":"$GName"}}])]
        all_teams_dict = {name:{} for name in all_teams_names}
        features = Features(self)
        for team in all_teams_dict:
            res_by_all, res_by_fix, res_by_non_avg = features.create_features(team)
            for fix in sorted(res_by_all):
                if fix == 1:
                    continue
                all_teams_dict[team][fix] = [res_by_all[fix][k] for k in sorted(res_by_all[fix])]
            for fix in sorted(res_by_fix):
                if fix == 1:
                    continue
                all_teams_dict[team][fix] += [res_by_fix[fix][k] for k in sorted(res_by_fix[fix])]
            for fix in sorted(res_by_non_avg):
                if fix == 1:
                    continue
                all_teams_dict[team][fix] += [res_by_non_avg[fix][k] for k in sorted(res_by_non_avg[fix])]
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

    
    
