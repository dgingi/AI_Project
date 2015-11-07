'''
Created on Nov 7, 2015

Implementations of different command line arguments parsers.

@author: root
'''
import argparse
from argparse import ArgumentParser

class CrawlerArgsParser(object):
    '''
    Arguments parser for the Selenium based web crawler module.
    '''
    leagues_links = {'Primer_League':"http://www.whoscored.com/Regions/252/Tournaments/2/England-Premier-League",
                     'Serie_A':"http://www.whoscored.com/Regions/108/Tournaments/5/Italy-Serie-A",
                     'La_Liga':"http://www.whoscored.com/Regions/206/Tournaments/4/Spain-La-Liga",
                     'Bundesliga':"http://www.whoscored.com/Regions/81/Tournaments/3/Germany-Bundesliga",
                     'Ligue_1':"http://www.whoscored.com/Regions/74/Tournaments/22/France-Ligue-1"
                     }


    def __init__(self):
        
    
        self.multi = False
        self.range_kwargs= []
        self.parser = ArgumentParser(description='Crawl whoscored.com for the specified league and years.')
        self.parser.add_argument('league', metavar='League', type=str,
                           help='A league to parse. The leagues are: '+', '.join(self.leagues_links.keys()),
                           choices=self.leagues_links.keys())
        self.parser.add_argument('year', metavar='Years',type=str, nargs='?',
                            help='A year to parse. Valid years: '+', '.join([str(i) for i in range(2010,2015)]),
                            default=str(max(range(2010,2015))),\
                            choices=self._default_ranges())
     
     
    def _default_ranges(self):
        return [str(i) for i in range(2010,2015)]+['-'.join([str(i),str(j)]) for i in range(2010,2015) for j in range(2010,2015) if i<j]

    def parse(self):
        args = self.parser.parse_args()
        self.LEAGUE_NAME = vars(args)['league']
        self.kwargs={}
        self.kwargs['league'] = self.leagues_links[vars(args)['league']]
        if '-' in vars(args)['year']:
            self.multi = True
            start , end = tuple(vars(args)['year'].split('-'))
            for year in range(int(start),int(end)+1):
                self.kwargs['year'] = int(year)
                self.range_kwargs.append(dict(self.kwargs))
        else:
            self.kwargs['year'] = int(vars(args)['year'])
        