from time import clock
from pymongo import MongoClient
import numpy as np
from pickle import load,dump
import os
import copy

MIN_YEAR = 2010
MAX_YEAR = 2015

class Features():
    def __init__(self,data,year):
        self.data = data
        self.col = data[year]
        self.non_avg_keys = ["Position","PName","GName","Result","HA","_id","Tag","VS","Goals","Fix"]
        self.d_pos = ["GK","DR","DL","DC","DMC","DML","DMR","MR","MC","ML"]
        self.sd_pos = ["DR","DL","DC"]
        self.m_pos = ["DMC","DML","DMR","MR","MC","ML","AMC","AML","AMR"]
        self.a_pos = ["FW","AR","AL","AC","AMC","AML","AMR"]
        self.sa_pos = ["FW","AR","AL","AC"]
        self.o_pos = ["Sub"]
        self.curr_year = int(year)
        self.prev_year = self.curr_year - 1

    def create_features(self,t_name,lookback=5):
        max_fix = max([g["Fix"] for g in self.col.find({"GName":t_name})])
        res_by_all = {i:self.create_avg_up_to("by_all_fix",t_name, i, lookback) for i in range(1,max_fix+1)}
        res_by_fix = {i:self.create_avg_up_to("by_fix",t_name, i, lookback) for i in range(1,max_fix+1)}
        res_by_fix_sum = {i:self.create_avg_up_to("by_fixSum",t_name, i, lookback) for i in range(1,max_fix+1)}
        res_by_non_avg = {i:self.create_avg_of_non_avg_f(t_name, i, lookback) for i in range(1,max_fix+1)}
        return res_by_all,res_by_fix,res_by_non_avg,res_by_fix_sum
           
    def get_curr_HA(self,t_name,fix):
        return self.col.find_one({"GName":t_name,"Fix":fix})["HA"]
    
    def get_curr_VS(self,t_name,fix):
        return self.col.find_one({"GName":t_name,"Fix":fix})["VS"]
    
    def get_agg_size(self,agg):
                agg_size = 0
                for cursor in agg:
                    agg_size += 1
                return 1 if agg_size == 0 else agg_size
    
    def check_for_history(self,fix,lookback,need_history):
        if fix == 1 and (str(self.prev_year) not in self.data.keys()):
            return False,need_history
        if fix == 1 and (self.data[str(self.prev_year)].count() == 0):
            return False,need_history
        if fix - lookback <= 0 and str(self.prev_year) in self.data.keys():
            if self.data[str(self.prev_year)].count() != 0:
                need_history = True
        return True,need_history
    
    def get_history(self,res,t_name,fix,lookback,HA_list,pos_list,group_q,curr_key,curr_feat,func):
        temp_f = Features(self.data,str(self.prev_year))
        try:
            max_fix = max([g["Fix"] for g in temp_f.col.find({"GName":t_name})])
        except Exception,e:
            return 0
        diff = (fix-lookback)*(-1) + 1
        pipe = [{"$match":{"GName":t_name,"Touches":{"$gt":0},"Fix":{"$lte":max_fix,"$gt":max_fix-diff},"HA":{"$in":HA_list}}}]
        if pos_list:
                    pipe[0]["$match"]["Position"] = {"$in":pos_list}
        pipe += [group_q]
        agg = temp_f.col.aggregate(pipe)
        num_of_games = self.get_agg_size(agg)
        agg = temp_f.col.aggregate(pipe)
        orig_curr_key = curr_key 
        for cursor in agg:
            for key in cursor:
                if key!="_id" or curr_feat in ["GR","SR","PR","SSR"]:
                    if orig_curr_key == "all":
                        curr_key=key
                    if curr_feat == "GR":
                        res[curr_key] += func(cursor[key]["HA"],cursor[key]["Result"])
                    else:
                        res[curr_key] += func(cursor,key)
        return num_of_games
        
    def create_avg_of_non_avg_f(self,t_name,fix,lookback=5):
        
        def update_avg_goals_scored(res,t_name,fix,by_loc,HA_list,lookback=5):
            need_history = False
            his_num_of_games = 0
            break_or_continue,need_history = self.check_for_history(fix, lookback, need_history)
            if not break_or_continue:
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
            
            if need_history:
                his_num_of_games = self.get_history(res, t_name, fix, lookback, HA_list, [], group_q, "avg_Goals_by_fix"+by_loc, "GS", lambda c,k: c[k])
            res["avg_Goals_by_fix"+by_loc] /= (num_of_games+his_num_of_games)
        
        def update_avg_received_goals(res,t_name,fix,by_loc,HA_list,lookback=5):
            
            def select_recieved_goals(HA,result):
                if HA=="home":
                    return int(result[1])
                else:
                    return int(result[0])
            
            need_history = False
            his_num_of_games = 0
            break_or_continue,need_history = self.check_for_history(fix, lookback, need_history)
            if not break_or_continue:
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
            
            if need_history:
                his_num_of_games = self.get_history(res, t_name, fix, lookback, HA_list, [], group_q, "avg_received_Goals_by_fix"+by_loc, "GR", select_recieved_goals)
            res["avg_received_Goals_by_fix"+by_loc] /= (num_of_games+his_num_of_games)
        
        def update_avg_success_rate(res,t_name,fix,by_loc,HA_list,lookback=5):
            need_history = False
            his_num_of_games = 0
            break_or_continue,need_history = self.check_for_history(fix, lookback, need_history)
            if not break_or_continue:
                return
            pipe = [{"$match":{"GName":t_name,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback},"HA":{"$in":HA_list}}}]
            group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix","Tag":"$Tag"}}}
            pipe += [group_q]
            res["avg_Success_rate"+by_loc] = 0.0
            agg = self.col.aggregate(pipe)
            num_of_games = self.get_agg_size(agg)
            agg = self.col.aggregate(pipe)
            for cursor in agg:
                for key in cursor:
                    res["avg_Success_rate"+by_loc] += 1 if cursor[key]["Tag"]==1 else 0 
            
            if need_history:
                his_num_of_games = self.get_history(res, t_name, fix, lookback, HA_list, [], group_q, "avg_Success_rate"+by_loc, "SR", lambda c,k: 1 if c[k]["Tag"]==1 else 0)
            res["avg_Success_rate"+by_loc] /= (num_of_games+his_num_of_games)
            res["avg_Success_rate"+by_loc] *= 100
        
        def update_avg_specific_success_rate(res,t_name,vs_t_name,fix,by_loc,HA_list,lookback=5):
            need_history = False
            his_num_of_games = 0
            break_or_continue,need_history = self.check_for_history(fix, lookback, need_history)
            if not break_or_continue:
                return
            pipe = [{"$match":{"GName":t_name,"VS":vs_t_name,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback},"HA":{"$in":HA_list}}}]
            group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix","Tag":"$Tag"}}}
            pipe += [group_q]
            res["avg_Specific_Success_rate"+by_loc] = 0.0
            agg = self.col.aggregate(pipe)
            num_of_games = self.get_agg_size(agg)
            agg = self.col.aggregate(pipe)
            for cursor in agg:
                for key in cursor:
                    res["avg_Specific_Success_rate"+by_loc] += 1 if cursor[key]["Tag"]==1 else 0 
            
            if need_history:
                his_num_of_games = self.get_history(res, t_name, fix, lookback, HA_list, [], group_q, "avg_Specific_Success_rate"+by_loc, "SSR", lambda c,k: 1 if c[k]["Tag"]==1 else 0)
            res["avg_Specific_Success_rate"+by_loc] /= (num_of_games+his_num_of_games)
            res["avg_Specific_Success_rate"+by_loc] *= 100
              
        def update_avg_possesion_rate(res,t_name,fix,by_loc,HA_list,lookback=5):   
            need_history = False
            his_num_of_games = 0
            break_or_continue,need_history = self.check_for_history(fix, lookback, need_history)
            if not break_or_continue:
                return
            pipe = [{"$match":{"GName":t_name,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback},"HA":{"$in":HA_list}}}]
            group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix","Possession":"$Possession"}}}
            pipe += [group_q]
            res["avg_Possession_rate"+by_loc] = 0.0
            agg = self.col.aggregate(pipe)
            num_of_games = self.get_agg_size(agg)
            agg = self.col.aggregate(pipe)
            for cursor in agg:
                for key in cursor:
                    res["avg_Possession_rate"+by_loc] += cursor[key]["Possession"] 
            
            if need_history:
                his_num_of_games = self.get_history(res, t_name, fix, lookback, HA_list, [], group_q, "avg_Possession_rate"+by_loc, "PR", lambda c,k: c[k]["Possession"])
            res["avg_Success_rate"+by_loc] /= (num_of_games+his_num_of_games)
            
        res = {}
        curr_HA = self.get_curr_HA(t_name, fix) 
        curr_VS = self.get_curr_VS(t_name, fix)
        
        for func in [update_avg_goals_scored,update_avg_received_goals,update_avg_success_rate,update_avg_possesion_rate]:
            func(res,t_name, fix, "_by_all_HA", ["home","away"], lookback)
            func(res,t_name, fix, "_by_"+curr_HA, [curr_HA], lookback)
        
        update_avg_specific_success_rate(res, t_name, curr_VS, fix, "_by_all_HA", ["home","away"], lookback)
        update_avg_specific_success_rate(res, t_name, curr_VS, fix, "_by_"+curr_HA, [curr_HA], lookback)
        
        return res
        
    def create_avg_up_to(self,by_avg,t_name,fix,lookback=5):
        
        def decide_pos_list(str):
            pos = str.split('_')[1]
            if pos == 'all':
                return []
            elif pos == 'def':
                return self.d_pos
            elif pos == 'sdef':
                return self.sd_pos
            elif pos == 'mid':
                return self.m_pos
            elif pos == 'att':
                return self.a_pos
            elif pos == 'satt':
                return self.sa_pos
            else:
                return self.o_pos
        
        def make_pos_list(str):
            if str == "by_all_fix":
                return ["by_all_pos","by_sub_pos","by_mid_pos","by_satt_pos"]
            else:
                return ["by_all_pos","by_def_pos","by_att_pos","by_sub_pos","by_mid_pos","by_satt_pos","by_sdef_pos"]
           
        def update_res(res,t_name,fix,lookback,by_avg,by_pos,by_loc,HA_list,pos_list):
            
            def create_pipe(res,t_name,fix,lookback,by_avg,by_pos,by_loc,HA_list,pos_list):
                
                def all_features(res,by_avg,by_pos,by_loc,group_q):
                    temp_group_q = dict(group_q)
                    for key in self.col.find_one().keys():
                        if key in self.non_avg_keys:
                            continue
                        index = "$"+key
                        if by_avg!="by_fixSum":
                            temp = ("avg_"+key+"_"+by_avg+"_"+by_pos+"_"+by_loc,{"$avg":index})
                        else:
                            temp = ("avg_"+key+"_"+by_avg+"_"+by_pos+"_"+by_loc,{"$sum":index})
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
                new_group_q = all_features(res,by_avg,by_pos,by_loc, group_q) 
                pipe += [new_group_q]
                return pipe,new_group_q
            
            need_history = False
            his_num_of_games = 0
            break_or_continue,need_history = self.check_for_history(fix, lookback, need_history)
            if not break_or_continue:
                return
            
            temp_res = {}
            pipe,group_q = create_pipe(temp_res, t_name, fix, lookback,by_avg, by_pos, by_loc, HA_list, pos_list)
            agg = self.col.aggregate(pipe)
            num_of_games = self.get_agg_size(agg)
            agg = self.col.aggregate(pipe)
            for cursor in agg:
                for key in cursor:
                    if key != "_id":
                        temp_res[key]+=cursor[key]
            
            if need_history:
                his_num_of_games = self.get_history(temp_res, t_name, fix, lookback, HA_list, pos_list, group_q, "all", "AF", lambda c,k: c[k])
            if by_avg != "by_all_fix":
                for key in temp_res:
                    temp_res[key] /= (num_of_games+his_num_of_games)
            if by_avg == "by_all_fix" and need_history and his_num_of_games!= 0:
                for key in temp_res:
                    temp_res[key] /= 2
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
    def __init__(self,league):
        self.client = MongoClient() #TODO: remote DB
        self.DB = self.client[league]
        self.cols = {str(year):self.DB[str(year)] for year in range(MIN_YEAR,MAX_YEAR)}
        self.league = league
    
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
    
    def insert_to_db(self,data,year):
        if self.cols[year].count() != 0:
            self.drop(year)
        self.cols[year].insert(self.explode(self.convert(data)))
        
    def drop(self,year=None):
        if year:
            self.client[self.league].drop_collection(year)
        else:
            self.client.drop_database(self.league)
           
    def create_examples(self,year,lookback=5):
        def update_all_teams_dict(res,all_teams_dict,team,first):
            for fix in sorted(res):
                if fix == 1 and res[fix] == {}:
                    all_teams_dict[team][fix] = []
                    continue
                if first:
                    all_teams_dict[team][fix] = [res[fix][k] for k in sorted(res[fix])]
                else:
                    all_teams_dict[team][fix] += [res[fix][k] for k in sorted(res[fix])]
        
        all_teams_names = [g['_id'] for g in self.cols[year].aggregate([{"$group":{"_id":"$GName"}}])]
        all_teams_dict = {name:{} for name in all_teams_names}
        features = Features(self.cols,year)
        for team in all_teams_dict:
            res_by_all, res_by_fix, res_by_non_avg, res_by_fix_sum = features.create_features(team,lookback)
            update_all_teams_dict(res_by_all, all_teams_dict, team, True)
            update_all_teams_dict(res_by_fix, all_teams_dict, team, False)
            update_all_teams_dict(res_by_fix_sum, all_teams_dict, team, False)
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
                    examples += [np.array(all_teams_dict[team][fix])-np.array(all_teams_dict[curr_game["VS"]][vs_curr_fix])]
                    tags += [curr_game["Tag"]]
        return examples,tags
        

class EXHandler():
    def __init__(self,league):
        self.league = league
        self.DBH = DBHandler(self.league)
        return

    def get_features_names(self):
        D = DBHandler(self.league)
        team = [g['_id'] for g in D.cols["2012"].aggregate([{"$group":{"_id":"$GName"}}])][0]
        res_by_all, res_by_fix, res_by_non_avg, res_by_fix_sum = Features(D.cols,"2012").create_features(team)
        features_names = [k for k in sorted(res_by_all[15])]
        features_names += [k for k in sorted(res_by_fix[15])]
        features_names += [k for k in sorted(res_by_fix_sum[15])]
        features_names += [k for k in sorted(res_by_non_avg[15])]
        return features_names
    
    def get(self,year=None):
        examples = []
        tags = []
        if not year:
            all_years = range(MIN_YEAR,MAX_YEAR)
        else:
            all_years = [year]
        for curr_year in range(MIN_YEAR,MAX_YEAR):
            path = self.league+"-"+str(curr_year)+"\\"+self.league+"-"+str(curr_year)
            if self.DBH.cols[str(curr_year)].count() != 0:
                if os.path.exists(path+"_E.pckl"):
                    with open(path+"_E.pckl",'r') as res:
                        temp_e = load(res)
                        examples += temp_e
                    with open(path+"_T.pckl",'r') as res:
                        temp_t = load(res)
                        tags += temp_t
                else:
                    temp_e,temp_t = self.DBH.create_examples(str(curr_year))
                    examples += temp_e
                    tags += temp_t
                    with open(path+"_E.pckl",'w') as res:
                        dump(temp_e,res)
                    with open(path+"_T.pckl",'w') as res:
                        dump(temp_t,res)
            elif os.path.exists(path+"-May.pckl"):
                with open(path+"-May.pckl",'r') as res:
                    data = load(res)
                    self.DBH.insert_to_db(data, str(curr_year))
                    temp_e,temp_t = self.DBH.create_examples(str(curr_year))
                    examples += temp_e
                    tags += temp_t
                    with open(path+"_E.pckl",'w') as res:
                        dump(temp_e,res)
                    with open(path+"_T.pckl",'w') as res:
                        dump(temp_t,res)
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

def PlotGraph(x_data,y_data,y_max_range,x_title,y_title,graph_name,plot_type):
    import matplotlib.pyplot as plt
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(graph_name)
    plt.plot(x_data,y_data,plot_type)
    plt.axis([0,max(x_data)+1,0,y_max_range])
    plt.savefig("tests/pictures/"+graph_name+".png")
    plt.close()

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

    
    
