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
    
   
        
"""res = []
line = {}
for team in data:
    for fix in data[team]:
        for player in data[team][fix]['Players']:
            for table in data[team][fix]['Players'][player]:
                for key in data[team][fix]['Players'][player][table]:
                    line[key] = data[team][fix]['Players'][player][table][key]
            else:
                line['PName'] = player
                line['GName'] = team
                line['Fix'] = fix
                line['HA'] = data[team][fix]['HA']
                line['Result'] = data[team][fix]['Result']
                res.append(line)"""
                
            

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