from abstract_search import *
from random import shuffle,randint,random
import numpy as np
from sklearn import tree
import copy

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
    check_borders(new_state, "max_features", 1, len(new_state.examples[0]))
    return new_state

def funcMaxDAux(i):
    return lambda state: funcMaxD(state,i)

def funcMaxD(state,i):
    new_state = copy.deepcopy(state)
    new_state.data["max_depth"] += i
    check_borders(new_state, "max_depth", 1, len(new_state.examples))
    return new_state

def funcMinSSAux(i):
    return lambda state: funcMinSS(state,i)

def funcMinSS(state,i):
    new_state = copy.deepcopy(state)
    new_state.data["min_samples_split"] += i
    check_borders(new_state, "min_samples_split", 2, len(new_state.examples))
    return new_state
    
def funcMinSLAux(i):
    return lambda state: funcMinSL(state, i)

def funcMinSL(state,i):
    new_state = copy.deepcopy(state)
    new_state.data["min_samples_leaf"] += i
    check_borders(new_state, "min_samples_leaf", 1, len(new_state.examples))
    return new_state

    
class LearningState(SearchState):
    def __init__(self,data,examples,tags):
        legal_operators = [funcCriterion,funcSplitter]
        legal_operators += [funcMaxDAux(i) for i in range(1,15)]
        legal_operators += [funcMaxDAux(-i) for i in range(1,15)]
        legal_operators += [funcMaxFAux(i) for i in range(1,15)]
        legal_operators += [funcMaxFAux(-i) for i in range(1,15)]
        legal_operators += [funcMinSLAux(i) for i in range(1,15)]
        legal_operators += [funcMinSLAux(-i) for i in range(1,15)]
        legal_operators += [funcMinSSAux(i) for i in range(1,15)]
        legal_operators += [funcMinSSAux(-i) for i in range(1,15)]
        SearchState.__init__(self,legal_operators)
        self.examples = examples
        self.tags = tags
        self.data = data
    
    def evaluate(self, evaluation_set, evaluation_set_labels,avg_amount=20):
        avg_succ = 0.0
        for i in range(avg_amount):
            clf = tree.DecisionTreeClassifier(criterion=self.data["criterion"],splitter=self.data["splitter"],max_features=self.data["max_features"],max_depth=self.data["max_depth"],min_samples_split=self.data["min_samples_split"],min_samples_leaf=self.data["min_samples_leaf"])
            clf = clf.fit(self.examples,self.tags)
            res = clf.predict(evaluation_set)
            sum = 0
            for j in range(len(res)):
                if res[j] == evaluation_set_labels[j]:
                    sum += 1
            avg_succ += float(sum)/len(res)
        return avg_succ/avg_amount


class FirstChoiceLocalSearch(LocalSearch):
    def __init__(self,examples,tags):
        data = {"criterion":"gini","splitter":"random","max_features":len(examples[0]),"max_depth":len(examples),"min_samples_split":2,"min_samples_leaf":1}
        starting_state = LearningState(data,examples,tags)
        LocalSearch.__init__(self,starting_state)
        self.sideSteps = 30
    
    def make_random_state(self,state,array):
        new_data = {}
        prev_state = True
        while prev_state:
            found_prev = False
            new_data["criterion"] = "gini" if random() <= 0.5 else "entropy"
            new_data["splitter"] = "random" if random() <= 0.5 else "best"
            new_data["max_features"] = randint(1,len(state.examples[0]))
            new_data["max_depth"] =  len(state.examples) if random() <= 0.5 else randint(1,len(state.examples))
            new_data["min_samples_split"] = randint(2,len(state.examples))
            new_data["min_samples_leaf"] = randint(1,len(state.examples))
            for i in range(len(array)):
                if array[i][1] == new_data:
                    found_prev = True
                    break
            if not found_prev:
                prev_state = False
        new_state = LearningState(new_data,state.examples,state.tags)
        return new_state
        
    
    def search(self, evaluation_set, evaluation_set_labels):
        current = self._current_state
        output_array = []
        random_array = []
        i = 1
        random_start = 1
        first_state = True
        
        while True:
            improved = False
            next_states = current.get_next_states()
            shuffle(next_states)
            if first_state:
                cRes = current.evaluate(evaluation_set, evaluation_set_labels)
            output_array += [(i,cRes)]
            i += 1
            
            while not improved and self.sideSteps >= 0:
                try:
                    new_state = next_states.pop(0)[0]
                except Exception,e:
                    break
                nRes = new_state.evaluate(evaluation_set, evaluation_set_labels)
                if nRes >= cRes :
                    if nRes == cRes:
                        self.sideSteps -= 1
                    if nRes > cRes:
                        self.sideSteps = 30
                    
                    current = new_state
                    cRes = nRes
                    first_state = False
                    improved = True
            
            if not improved:
                if random_start <= 10:
                    random_array += [(cRes,current)]
                    current = self.make_random_state(current,output_array)
                    random_start += 1
                    first_state = True
                    self.sideSteps = 30
                else:
                    return max(random_array)[1],max(random_array)[0],output_array



