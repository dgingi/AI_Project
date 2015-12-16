'''
Created on Nov 27, 2015

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

def retry(f):
    '''
    Decorator for retrying a function.
    
    Usage:
    
    @retry
    def some_function(*args,**kwargs):
    '''
    def wrap(*x,**d):
        NUM_RETRIES = 3
        for i in range(NUM_RETRIES):
            try:
                res = f(*x,**d)
                return res
            except Exception as e:
                logging.error(str(e))
        else:
            return None
    return wrap

def force(f):
    '''
    Decorator for forcing a function to success in case of webdriver failing.
    
    Usage:
    
    @force
    def some_function(*args,**kwargs):
    '''
    def wrap(*x,**d):
        while True:
            try:
                res = f(*x,**d)
                return res
            except Exception as e:
                logging.error(str(e))
                x[0].restart_driver()
    return wrap