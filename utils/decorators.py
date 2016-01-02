'''
Decorators and various generic utilities for the project.

@author: Ory Jonay
'''

from time import clock,sleep
import logging

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

def retry(restart=False,num_retries=3):
    '''
    Decorator for retrying a function in case of an exception.
    
    Usage:
    
    @retry
    def some_function(*args,**kwargs):
    '''
    def _wrap(f):
        def wrap(*x,**d):
            for i in range(num_retries):
                try:
                    res = f(*x,**d)
                    return res
                except Exception as e:
                    logging.error(str(e))
                    if restart:
                        x[0].restart_driver()
            else:
                logging.critical('Exceeded number of retries')
                try:
                    logging.critical('Skipping game- %s VS %s'%(x[1]['home'],x[1]['away']))
                except:
                    pass
                return None
        return wrap
    return _wrap


    