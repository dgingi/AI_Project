from abstract_search import *
from random import shuffle,randint,random
import numpy as np
from sklearn import tree
 
def funcCriterion(state):
    new_data = state.data.copy()
    new_state = LearningState(new_data,state.examples,state.tags)
    if state.data["criterion"]=="gini":
        new_state.data["criterion"] = "entropy"
    else:
        new_state.data["criterion"] = "gini"
    return new_state 

def funcSplitter(state):
    new_data = state.data.copy()
    new_state = LearningState(new_data,state.examples,state.tags)
    if state.data["splitter"]=="best":
        new_state.data["splitter"] = "random"
    else:
        new_state.data["splitter"] = "best"
    return new_state

def funcMaxFAux(i):
    return lambda state: funcMaxF(state, i)

def funcMaxF(state,i):
    new_data = state.data.copy()
    new_state = LearningState(new_data,state.examples,state.tags)
    new_state.data["max_features"] = i
    return new_state


def funcMaxDAux(i):
    return lambda state: funcMaxD(state,i)

def funcMaxD(state,i):
    new_data = state.data.copy()
    new_state = LearningState(new_data,state.examples,state.tags)
    new_state.data["max_depth"] = i
    return new_state

def funcMinSSAux(i):
    return lambda state: funcMinSS(state,i)

def funcMinSS(state,i):
    new_data = state.data.copy()
    new_state = LearningState(new_data,state.examples,state.tags)
    new_state.data["min_samples_split"] = i
    return new_state
    

def funcMinSLAux(i):
    return lambda state: funcMinSL(state, i)

def funcMinSL(state,i):
    new_data = state.data.copy()
    new_state = LearningState(new_data,state.examples,state.tags)
    new_state.data["min_samples_leaf"] = i
    return new_state

    
class LearningState(SearchState):
    def __init__(self,data,examples,tags):
        legal_operators = [funcCriterion]#,funcSplitter]#,funcMaxDRand,funcMaxFRand,funcMinSLRand,funcMinSSRand]
        legal_operators += [funcMaxDAux(i) for i in range(1,len(examples))]
        legal_operators += [funcMaxFAux(i) for i in range(1,len(examples[0]))]
        legal_operators += [funcMinSLAux(i) for i in range(1,len(examples))]
        legal_operators += [funcMinSSAux(i) for i in range(2,len(examples))]
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
            new_data["splitter"] = "random" if random() <= 0.5 else "random"
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
            while not improved:
                new_state = next_states.pop(0)[0]
                nRes = new_state.evaluate(evaluation_set, evaluation_set_labels)
                cRes = current.evaluate(evaluation_set, evaluation_set_labels)
                if first_state:
                    output_array += [(i,cRes)]
                    print cRes
                    i += 1
                    first_state = False
                if  nRes >= cRes :
                    print "found better",cRes,nRes
                    if nRes == cRes and self.sideSteps > 0:
                        self.sideSteps -= 1
                        improved = True
                        current = new_state
                        output_array += [(i,nRes)]
                        print nRes
                        i += 1
                    if nRes > cRes:
                        self.sideSteps = 30
                        improved = True
                        first_state = True
            if not improved and random_start <=0:
                random_array += [(output_array[-1][1],current)]
                print "restart2",random_array[-1][0]
                current = self.make_random_state(current,output_array)
                random_start += 1
                first_state = True
                improved = False
            if not improved and random_start > 10:
                return max(random_array)[1],max(random_array)[0],output_array




