from datetime import datetime,timedelta
import logging
from contextlib import contextmanager
import os
from utils.constants import PROJECT_ROOT

@contextmanager
def move_to_root_dir():
    """
    Context manager to move automatically move to the project root directory in case we are not running from it.
    
    Usage::
    
        with move_to_root_dir():
            do_some_stuff()
    """
    if os.getcwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)
    yield
    if os.getcwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)


def timed(f):
    """
    Decorator for printing the timing of functions
    
    Usage:: 
    
        @timed
        def some_funcion(args...):
    """
    
    def wrap(*x, **d):
        start = datetime.now()
        res = f(*x, **d)
        print 'Function "{0}" ran for {1} (Hours:Minutes:Seconds)'.format(f.__name__,str(timedelta(seconds=(datetime.now() - start).total_seconds()))) 
        return res
    return wrap

def retry(restart=False,num_retries=3):
    """
    Decorator for retrying a function in case of an exception.
    
    Usage::
    
        @retry
        def some_function(*args,**kwargs):
    """
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


    