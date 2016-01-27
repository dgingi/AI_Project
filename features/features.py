'''
Created on Jan 19, 2016

@author: dror
'''

from utils.decorators import timed
from utils.constants import MIN_YEAR
import logging
from os import path

class Features():
    """This class handles the creation of the features for the classifier. 

    """
    def __init__(self,data,year,league):
        """Init method - used to create new instance of Features.

        Args:
           data(dictionary):  A dictionary of {"year": all data for this year}.
           
           year(str): This is the year that we want to create the features for.
        
        This function initializes several lists of positions, such as:
        
        attack position list = ["FW","AR","AL","AC","AMC","AML","AMR"]
        
        it also saves the current year and previous year as an int.
        """
        self.col = data
        
        self.non_avg_keys = ["Position","PName","GName","Result","HA","_id","Tag","VS","Goals","Fix"]
        self.all_keys = ["PA%","AerialsWon","Touches","Passes","Crosses","AccCrosses","LB","AccLB","ThB","AccThB"]
        self.att_keys = ["Shots","ShotsOT","KeyPasses","Dribbles","Fouled","Offsides","Disp","UnsTouches"]
        self.def_keys = ["TotalTackles","Interceptions","Clearances","BlockedShots","Fouls"]
        
        self.d_pos = ["DR","DL","DC","DMC","DML","DMR","MR","MC","ML","Sub"]
        self.a_pos = ["FW","AR","AL","AC","AMC","AML","AMR","MR","MC","ML","Sub"]
        
        self.curr_year = int(year)
        self.prev_year = self.curr_year - 1
        
        self.league = league
        
        logging.basicConfig(filename=path.join('logs','features-%s-%s.log'%(league,year)),format='%(levelname)s: %(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M %p',level=logging.INFO)
    
    @timed
    def create_features(self,t_name,lookback=15):
        """A method to create features for a team, according to the lookback.

        Args:
           * t_name (str):  The name of the team we want to make the features for.
           
           * lookback(int): The amount of lookback to make all the aggregation.
        
        Returns:
        
        This method returns 4 different dictionaries.
        
        Each dicitionary is in the form of --> {fix_num(int) : {key:key_result} }
        
        There are 4 dictionaries:
            1. res_by_all: this dicitionary has features that are aggregated and avareged first by each fix and then the avg of all fixtures.
            
            2. res_by_fix: this dicitionary has features that are aggregated and avareged by all fixtures togther.
            
            3. res_by_non_avg: this dicitionary has features that are aggregated and avareged by summing up all values and avarging by amount of players.
            
            4. res_by_fix_sum: this dicitionary has features that are aggregated and avareged by first summing all fixtures toghter and making an avarege by amount of fixtures.
            
        Example of use:
        
        ::
            
             res_by_all,res_by_fix,res_by_non_avg,res_by_fix_sum = create_features("Chelsea",15)
             res_by_non_avg[3]["avg_Goals_Scored"] = 2.7
             
        """
        max_fix = max([g["Fix"] for g in self.col.find({"GName":t_name,"Year":self.curr_year})])
        res_by_all = {i:self.create_avg_up_to(t_name, i, lookback) for i in range(1,max_fix+1)}
        res_by_non_avg = {i:self.create_avg_of_non_avg_f(t_name, i, lookback) for i in range(1,max_fix+1)}
        return res_by_all,res_by_non_avg
           
    def get_curr_HA(self,t_name,fix):
        """This method returns whether t_name played as home / away in the game fix. 

        Args:
           * t_name (str):  The name of the team we want to find the current location of the game.
           
           * fix(int): The fix that we want to search.
        
        Returns:
        
        This function returns the tag (home / away) of the requested team for the requested game.
        
        Example of use:
        
        ::
            
             curr_HA = get_curr_HA("Chelsea",3)
             curr_HA
             --> away
             curr_HA = get_curr_HA("liverpool",3)
             curr_HA
             --> home
             
        """
        return self.col.find_one({"GName":t_name,"Fix":fix,"Year":self.curr_year})["HA"]
    
    def get_curr_VS(self,t_name,fix):
        """This function returns the matching team name that played against Team=(t_name) in Fix=fix.

        Args:
           * t_name (str):  The name of the group we want to find it's opponent.
           
           * fix(int): The fix that we want to search.
        
        Returns:
        
        This function returns the name of the relative team.
        
        Example of use:
        
        ::
            
             curr_VS = get_curr_HA("Chelsea",3)
             curr_VS
             --> Liverpool
             
        """
        return self.col.find_one({"GName":t_name,"Fix":fix,"Year":self.curr_year})["VS"]
    
    def get_agg_size(self,agg):
        """This function returns size of giving aggregation.

        Args:
           * agg(MongoCursor) : The aggregation we want to check.
        
        Returns:
        
        This function returns the size of the aggregation as int.
        
        Example of use:
        
        ::
            
             agg_size = get_agg_size(some_agg)
             agg_size
             --> 7
             
        """
        agg_size = 0
        for cursor in agg:
            agg_size += 1
        return agg_size
    
    def check_for_history(self,t_name,fix,lookback,need_history):
        """This function returns if for the current fixture that we are checking and according to the size of lookback 
        requested, we have enough games in this current year or not.
        
        for example if we are trying to search some features for fixture 5 and the lookback is 10 then we need to add 5 more games from the previous season.

        Args:
           * fix(int): The fix that we want to check.
           
           * lookback(int): The size of the requested lookback.
           
           * need_history(bool): This paramter will also save this function result for further usage.
        
        Returns:
        
        This function returns True if we need to add games from previous year for the requested aggregation and False otherwise.
        
        Example of use:
        
        ::
        
            need_history = False
            if check_for_history(5,7,need_history):
                ....
            else:
                ....
            need_history
            --> True
            
        """
        check_his = self.col.find_one({"Year":self.prev_year,"GName":t_name})
        if fix == 1 and not check_his:
            return False,need_history
        if fix - lookback <= 0 and check_his:
            need_history = True
        return True,need_history
    
    def select_recieved_goals(self,HA,result,bool_for_recieved):
            if HA=="home":
                if bool_for_recieved:
                    return int(result[1])
                else:
                    return int(result[0])
            else:
                if bool_for_recieved:
                    return int(result[0])
                else:
                    return int(result[1])
            
    def get_history(self,res,t_name,diff,year,HA_list,group_q,add_to_key,pos_list=[],all_feat=False):
        temp_f = Features(self.col,str(year),self.league)
        try:
            max_fix = max([g["Fix"] for g in temp_f.col.find({"GName":t_name,"Year":temp_f.curr_year})])
        except Exception,e:
            return 0
        pipe = [{"$match":{"GName":t_name,"Year":temp_f.curr_year,"Touches":{"$gt":0},"Fix":{"$lte":max_fix,"$gt":max_fix-diff},"HA":{"$in":HA_list}}}]
        if pos_list:
            pipe[0]["$match"]["Position"] = {"$in":pos_list}
        pipe += [group_q]
        agg = temp_f.col.aggregate(pipe)
        num_of_games = self.get_agg_size(agg)
        agg = temp_f.col.aggregate(pipe)
        for cursor in agg:
            for key in cursor:
                if key!="_id":
                    if not all_feat:
                        res["avg_Goals_by_fix"+add_to_key] += self.select_recieved_goals(cursor[key]["HA"], cursor[key]["Result"],False)
                    else:
                        res[key] += cursor[key]
                else:
                    if not all_feat:
                        res["avg_received_Goals_by_fix"+add_to_key] += self.select_recieved_goals(cursor[key]["HA"], cursor[key]["Result"],True)
                        res["avg_Success_rate"+add_to_key] += 1 if cursor[key]["Tag"]==1 else 0
                        res["avg_Possession_rate"+add_to_key] += cursor[key]["Possession"]
        return num_of_games,max_fix-diff
        
    def create_avg_of_non_avg_f(self,t_name,fix,lookback):
        
        def create_avg(res,t_name,fix,by_loc,HA_list,lookback,vs=""):
            
            need_history = False
            his_num_of_games = 0
            break_or_continue,need_history = self.check_for_history(t_name, fix, lookback, need_history)
            if not break_or_continue:
                return
            lookback = fix if not need_history else lookback  
            
            prev_pipe = [{"$match":{"GName":t_name,"Touches":{"$gt":0},"HA":{"$in":HA_list},"VS":vs,"Year":{"$in":[i for i in range(MIN_YEAR,self.curr_year)]}}}]
            pipe = [{"$match":{"GName":t_name,"Year":self.curr_year,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback},"HA":{"$in":HA_list}}}]
            if vs!= "":
                pipe[0]["$match"]["VS"] = vs           
            group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix","HA":"$HA","Result":"$Result","Tag":"$Tag","Possession":"$Possession"}}}
            prev_pipe += [group_q]
            pipe += [group_q]

            add_to_key = by_loc if vs == "" else by_loc + "_specific"
            
            res["avg_Goals_by_fix"+add_to_key] = 0.0
            res["avg_received_Goals_by_fix"+add_to_key] = 0.0
            res["avg_Success_rate"+add_to_key] = 0.0
            res["avg_Possession_rate"+add_to_key] = 0.0
            
            agg = self.col.aggregate(pipe)
            num_of_games = self.get_agg_size(agg)
            agg = self.col.aggregate(pipe) 
            agg_list = [agg]
            if vs != "":
                prev_agg = self.col.aggregate(prev_pipe)
                num_of_games += self.get_agg_size(prev_agg)
                prev_agg = self.col.aggregate(prev_pipe)
                agg_list += [prev_agg]
                
            for agg in agg_list:
                for cursor in agg:
                    for key in cursor:
                        if key!="_id":
                            res["avg_Goals_by_fix"+add_to_key] += self.select_recieved_goals(cursor[key]["HA"], cursor[key]["Result"],False)
                        else:
                            res["avg_received_Goals_by_fix"+add_to_key] += self.select_recieved_goals(cursor[key]["HA"], cursor[key]["Result"],True)
                            res["avg_Success_rate"+add_to_key] += 1 if cursor[key]["Tag"]==1 else 0
                            res["avg_Possession_rate"+add_to_key] += cursor[key]["Possession"]
                        
            if need_history and vs == "":
                diff = (fix-lookback)*(-1) + 1
                curr_year = self.curr_year
                his_num_of_games,more_his = self.get_history(res, t_name, diff, curr_year, HA_list, group_q, add_to_key)
                while more_his < 0:
                    curr_year -= 1
                    if curr_year < MIN_YEAR:
                        break
                    diff = -1 * more_his
                    temp_his_num_of_games,more_his = self.get_history(res, t_name, diff, curr_year, HA_list, group_q, add_to_key)
                    his_num_of_games += temp_his_num_of_games
                    
            if num_of_games == 0 and his_num_of_games == 0:
                logging.info("Got 0 in num_of_games and his_... with fix:%s, lookback:%s, team:%s, vs:%s"%(str(fix),str(lookback),t_name,vs))
                his_num_of_games = 1
            res["avg_Goals_by_fix"+add_to_key] /= (num_of_games+his_num_of_games)
            res["avg_received_Goals_by_fix"+add_to_key] /= (num_of_games+his_num_of_games)
            res["avg_Success_rate"+add_to_key] /= (num_of_games+his_num_of_games)
            res["avg_Success_rate"+add_to_key] *= 100
            res["avg_Possession_rate"+add_to_key] /= (num_of_games+his_num_of_games)
              
        res = {}
        curr_HA = self.get_curr_HA(t_name, fix) 
        curr_VS = self.get_curr_VS(t_name, fix)
        
        create_avg(res,t_name, fix, "_by_all_HA", ["home","away"], lookback)
        create_avg(res,t_name, fix, "_by_"+curr_HA, [curr_HA], lookback)
        
        create_avg(res,t_name, fix, "_by_all_HA", ["home","away"], lookback, curr_VS)
        create_avg(res,t_name, fix, "_by_"+curr_HA, [curr_HA], lookback, curr_VS)
        
        return res
        
    def create_avg_up_to(self,t_name,fix,lookback):
        
        def make_curr_lists(str):
            pos = str.split('_')[1]
            if pos == 'all':
                return [],self.all_keys
            elif pos == 'def':
                return self.d_pos,self.def_keys
            else:
                return self.a_pos,self.att_keys
          
        def update_res(res,t_name,fix,lookback,by_pos,by_loc,HA_list,pos_list,key_list):
            
            def create_pipe(res,t_name,fix,lookback,by_pos,by_loc,HA_list,pos_list,key_list):
                
                def all_features(res,by_pos,by_loc,group_q,key_list):
                    temp_group_q = dict(group_q)
                    for key in self.col.find_one().keys():
                        if key not in key_list:
                            continue
                        index = "$"+key
                        if key == "PA%":
                            temp = ("avg_"+key+"_"+by_pos+"_"+by_loc,{"$avg":index})
                        else:
                            temp = ("avg_"+key+"_"+by_pos+"_"+by_loc,{"$sum":index})
                        res[temp[0]] = 0.0
                        temp_group_q["$group"][temp[0]] = temp[1]
                    return temp_group_q
                
                pipe = [{"$match":{"GName":t_name,"Year":self.curr_year,"Touches":{"$gt":0},"Fix":{"$lt":fix,"$gte":fix-lookback},"HA":{"$in":HA_list}}}]
                if pos_list:
                    pipe[0]["$match"]["Position"] = {"$in":pos_list}
                group_q = {"$group":{"_id":{"GName":"$GName","Fix":"$Fix"}}}
                new_group_q = all_features(res,by_pos,by_loc, group_q,key_list) 
                pipe += [new_group_q]
                return pipe,new_group_q
            
            need_history = False
            his_num_of_games = 0
            break_or_continue,need_history = self.check_for_history(t_name, fix, lookback, need_history)
            if not break_or_continue:
                return
            lookback = fix if not need_history else lookback
            
            temp_res = {}
            pipe,group_q = create_pipe(temp_res, t_name, fix, lookback, by_pos, by_loc, HA_list, pos_list,key_list)
            
            agg = self.col.aggregate(pipe)
            num_of_games = self.get_agg_size(agg)
            agg = self.col.aggregate(pipe)
            for cursor in agg:
                for key in cursor:
                    if key != "_id":
                        temp_res[key] += cursor[key]
            
            if need_history:
                diff = (fix-lookback)*(-1) + 1
                curr_year = self.curr_year
                his_num_of_games,more_his = self.get_history(temp_res, t_name, diff, curr_year, HA_list, group_q, "add_to_key", pos_list, True)
                while more_his < 0:
                    curr_year -= 1
                    if curr_year < MIN_YEAR:
                        break
                    diff = -1 * more_his
                    temp_his_num_of_games,more_his = self.get_history(temp_res, t_name, diff, curr_year, HA_list, group_q, "add_to_key", pos_list, True)
                    his_num_of_games += temp_his_num_of_games
                    
                
            
            if num_of_games == 0 and his_num_of_games == 0:
                logging.info("Got 0 in num_of_games and his_... with fix:%s, lookback:%s, team:%s"%(str(fix),str(lookback),t_name))
                his_num_of_games = 1
            for key in temp_res:
                temp_res[key] /= (num_of_games+his_num_of_games)
            res.update(temp_res)
                                
        res = {}
        curr_HA = self.get_curr_HA(t_name, fix)
        pos_list = ["by_all_pos","by_def_pos","by_att_pos"]
        
        for pos in pos_list:
            curr_pos_list,curr_key_list = make_curr_lists(pos)
            update_res(res,t_name,fix,lookback,pos, "by_all_HA",["home","away"],curr_pos_list,curr_key_list)
            update_res(res,t_name,fix,lookback,pos, "by_"+curr_HA, [curr_HA],curr_pos_list,curr_key_list)
             
        return res   
    
    
                 
