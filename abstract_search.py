class SearchState(object):
    def __init__(self, legal_operators):
        '''Initialize search state.
        You should add some state to classes that inherit from this.
        '''
        self._legal_operators= legal_operators
    
    def evaluate(self, evaluation_set, evaluation_set_labels, *args, **kwargs):
        '''Evaluate state based on a labelled evaluation set.
        *args and **kwargs are optional parameters for the algorithm used
        '''
        raise NotImplementedError('Implement this in your class')
        
    def get_next_states(self):
        '''Implementation of the transitions for a state.
        This implementation requires your operators to receive a state and return a state (including operators).
        '''
        return [(state, op) for (state, op) in 
        zip([operator(self) for operator in self._legal_operators], self._legal_operators)]

class LocalSearch(object):
    def __init__(self, starting_state):
        '''Initialize a search problem with this starting_state.
        operators is is the list of operators used during the local search.
        '''
        self._current_state= starting_state
        self.operators= []
    
    def search(self, evaluation_set, evaluation_set_labels, *args, **kwargs): 
        '''Performs a local search based on a labelled evaluation set.
        *args and **kwargs are optional parameters for the algorithm used
        '''
        raise NotImplementedError('Implement this for your algorithm and search state')
