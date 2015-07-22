from time import clock
from pymongo import MongoClient

class DBHandler():
    def __init__(self):
        self.client = MongoClient() #TODO: remote DB
        self.DB = self.client.DB
        self.col = self.DB.col
    
    def convert(self,data):
        """
        Convert the crawler data keys into string for insertion into MongoDB
        """
        return {name:{str(i):data[name][i] for i in range(1,2*len(data.keys())-1)} for name in data.keys()}
    
    def explode(self,data):
        res = []
        for team in data:
            for fix in data[team]:
                for player in data[team][fix]['Players']:
                    line = {}
                    for table in data[team][fix]['Players'][player]:
                        for key in data[team][fix]['Players'][player][table]:
                            line[key] = data[team][fix]['Players'][player][table][key]
                    else:
                        line['PName'] = player
                        line['TName'] = team
                        line['Fix'] = int(fix)
                        line['HA'] = data[team][fix]['HA']
                        line['Result'] = data[team][fix]['Result']
                        line['Possession'] = data[team][fix]['Possession']
                        res.append(line)
        return res
    
    def insert_to_db(self,data):
        self.col.insert(self.explode(self.convert(data)))
        
        
                
            

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

pipe = [{"$match":{"GName":"Manchester United","Touches":{"$gt":0}}}]
z = {"$group":{"_id":"$Fix","tot_player":{"$sum":1}}}

z1 = {"$group":{"_id":"$Fix","tot_player":{"$sum":1}}}


def agg(data,z1):
for key in data.keys():
    y="$"+key
    x=("avg_"+key,{"$avg":y})
    z1["$group"][x[0]] = x[1]
    
    
pipe = [
        {"$match":{"GName":"Manchester United","Touches":{"$gt":0}}},
        {"$group":{"_id":"$Fix","tot_player":{"$sum":1},
                   "avg_goals":{"$avg":"$Goals"},
                   "avg_shots":{"$avg":"$Shots"},
                   "avg_shots_ot"}}]