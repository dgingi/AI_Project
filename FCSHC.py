from exhandler.exhandler import EXHandler
from abstract_search import *
from random import shuffle,randint,random
from sklearn import tree
import copy
from utils.constants import LEAGUES

def check_borders(state,key,lbord,ubord):
    if state.data[key] < lbord:
        state.data[key] = lbord
    if state.data[key] > ubord:
        state.data[key] = ubord

def funcCriterion(state):
    new_state = copy.deepcopy(state)
    if state.data["criterion"]=="gini":
        new_state.data["criterion"] = "entropy"
    else:
        new_state.data["criterion"] = "gini"
    return new_state 

def funcSplitter(state):
    new_state = copy.deepcopy(state)
    if state.data["splitter"]=="best":
        new_state.data["splitter"] = "random"
    else:
        new_state.data["splitter"] = "best"
    return new_state

def funcMaxFAux(i):
    return lambda state: funcMaxF(state, i)

def funcMaxF(state,i):
    new_state = copy.deepcopy(state)
    new_state.data["max_features"] += i
    check_borders(new_state, "max_features", 1, len(new_state.examples["Primer_League"][0]))
    return new_state

def funcMaxDAux(i):
    return lambda state: funcMaxD(state,i)

def funcMaxD(state,i):
    new_state = copy.deepcopy(state)
    new_state.data["max_depth"] += i
    check_borders(new_state, "max_depth", 1, len(new_state.examples["Primer_League"][0]))
    return new_state

def funcMinSSAux(i):
    return lambda state: funcMinSS(state,i)

def funcMinSS(state,i):
    new_state = copy.deepcopy(state)
    new_state.data["min_samples_split"] += i
    check_borders(new_state, "min_samples_split", 2, len(new_state.examples["Primer_League"]))
    return new_state
    
def funcMinSLAux(i):
    return lambda state: funcMinSL(state, i)

def funcMinSL(state,i):
    new_state = copy.deepcopy(state)
    new_state.data["min_samples_leaf"] += i
    check_borders(new_state, "min_samples_leaf", 1, len(new_state.examples["Primer_League"]))
    return new_state

    
class LearningState(SearchState):
    def __init__(self,data,ex_dict,ta_dict):
        legal_operators = [funcCriterion,funcSplitter]
        legal_operators += [funcMaxDAux(i) for i in range(1,10)]
        legal_operators += [funcMaxDAux(-i) for i in range(1,10)]
        legal_operators += [funcMaxFAux(i) for i in range(1,10)]
        legal_operators += [funcMaxFAux(-i) for i in range(1,10)]
        legal_operators += [funcMinSLAux(i) for i in range(1,10)]
        legal_operators += [funcMinSLAux(-i) for i in range(1,10)]
        legal_operators += [funcMinSSAux(i) for i in range(1,10)]
        legal_operators += [funcMinSSAux(-i) for i in range(1,10)]
        SearchState.__init__(self,legal_operators)
        self.examples = ex_dict
        self.tags = ta_dict
        self.data = data
    
    def evaluate2(self):
        avg_succ = 0.0
        for league in LEAGUES:
            clf = tree.DecisionTreeClassifier(criterion=self.data["criterion"],splitter=self.data["splitter"],max_features=self.data["max_features"],min_samples_split=self.data["min_samples_split"],min_samples_leaf=self.data["min_samples_leaf"],max_depth=self.data["max_depth"])
            X1 = []
            Y1 = []
            X2 = []
            Y2 = []
            for temp_league in LEAGUES:
                if temp_league != league:
                    X1 += self.examples[temp_league]
                    Y1 += self.tags[temp_league]
                else:
                    X2 = self.examples[temp_league]
                    Y2 = self.tags[temp_league] 
            clf = clf.fit(X1,Y1)
            res = clf.predict(X2)
            sum = 0
            for i in range(len(Y2)):
                if res[i] == Y2[i]:
                    sum+=1
            avg_succ += float(sum)/len(res)
        
        return avg_succ/len(LEAGUES)
    
    def evaluate(self):
        avg_succ = 0.0
        for league in LEAGUES:
            clf = tree.DecisionTreeClassifier(criterion=self.data["criterion"],splitter=self.data["splitter"],max_features=self.data["max_features"],min_samples_split=self.data["min_samples_split"],min_samples_leaf=self.data["min_samples_leaf"],max_depth=self.data["max_depth"])
            X1,Y1,X2,Y2 = EXHandler(league,False).split_to_train_and_test(self.examples[league], self.tags[league])
            clf = clf.fit(X1,Y1)
            res = clf.predict(X2)
            sum = 0
            for i in range(len(Y2)):
                if res[i] == Y2[i]:
                    sum+=1
            avg_succ += float(sum)/len(res)
        
        return avg_succ/len(LEAGUES)


class FirstChoiceLocalSearch(LocalSearch):
    def __init__(self):
        ex_dict = {}
        ta_dict = {}
        from exhandler.exhandler import EXHandler
        for league in LEAGUES:
            ex_dict[league], ta_dict[league] = EXHandler(league).get()
        
        data = {"criterion":"gini","splitter":"random","max_features":len(ex_dict["Primer_League"][0]),"min_samples_split":2,"min_samples_leaf":1,"max_depth":len(ex_dict["Primer_League"][0])}
        starting_state = LearningState(data,ex_dict,ta_dict)
        LocalSearch.__init__(self,starting_state)
        self.sideSteps = 30
        print "Finish Init"
    
    def make_random_state(self,state,array):
        new_data = {}
        prev_state = True
        while prev_state:
            print "looking for new random start"
            found_prev = False
            new_data["criterion"] = "gini" if random() <= 0.5 else "entropy"
            new_data["splitter"] = "random" if random() <= 0.5 else "best"
            new_data["max_features"] = randint(1,len(state.examples["Primer_League"][0]))
            new_data["max_depth"] =  len(state.examples["Primer_League"][0]) if random() <= 0.5 else randint(1,len(state.examples["Primer_League"][0]))
            new_data["min_samples_split"] = randint(2,len(state.examples["Primer_League"]))
            new_data["min_samples_leaf"] = randint(1,len(state.examples["Primer_League"]))
            for i in range(len(array)):
                if array[i][1] == new_data:
                    found_prev = True
                    break
            if not found_prev:
                prev_state = False
        new_state = LearningState(new_data,state.examples,state.tags)
        return new_state
        
    
    def search(self):
        print "Start Search"
        current = self._current_state
        output_array = []
        random_array = []
        i = 1
        random_start = 1
        first_state = True
        
        while True:
            print "start of loop"
            improved = False
            next_states = current.get_next_states()
            shuffle(next_states)
            if first_state:
                cRes = current.evaluate2()
            output_array += [(i,cRes)]
            i += 1
            
            while not improved and self.sideSteps >= 0:
                try:
                    print "Looking for next step"
                    new_state = next_states.pop(0)[0]
                except Exception,e:
                    break
                nRes = new_state.evaluate2()
                if nRes >= cRes :
                    if nRes == cRes:
                        print "Found Equal"
                        self.sideSteps -= 1
                    if nRes > cRes:
                        print "Found Better",cRes,nRes
                        self.sideSteps = 30
                    
                    current = new_state
                    cRes = nRes
                    first_state = False
                    improved = True
            
            print "Improved=",improved
            if not improved:
                print "finish looking all near stages no better"
                if random_start <= 10:
                    print "random"
                    random_array += [(cRes,current)]
                    current = self.make_random_state(current,output_array)
                    random_start += 1
                    first_state = True
                    self.sideSteps = 30
                else:
                    print "Finish-",random_array
                    return max(random_array)[1],max(random_array)[0],output_array



