from utils.constants import MIN_YEAR,MAX_YEAR
from argparse import ArgumentParser

class ExperimentArgsParser(object):
    """
    This class is incharge of parsering the arguments related to the experiments. 
    """
    experiments = ['Best_Params','AdaBoost','Best_Lookback','Best_Forest_Size','Bayes','Learning_Curve','Best_Proba','Final_Season','Default_Params']
    actions = ['run','report']
    
    def __init__(self):
        usage = '''experiments.py Output_Directory Experiment Run\Report  [-v {0,1,2}] [-o OUTFILE] [-h] 
example: 
$ experiments.py exp1 Best_Params report -v 1
                      '''
        self.parser = ArgumentParser(description='The different experiments for the project.',usage=usage)
        self.parser.add_argument('out_dir', metavar='Output_Directory', type=str,
                           help='The directory to store experiments or a directory that holds previous experiments.')
        self.parser.add_argument('exp', metavar='Experiment', type=str,
                           help='The experiment to run \ report. Choices are: '+' '.join(self.experiments),choices=self.experiments)
        self.parser.add_argument('action', metavar='Run\Report', type=str,choices=self.actions,help='Choices are: '+' / '.join(self.actions))
        self.parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1],default=0,
                    help="Increase the output verbosity for reporting an experiment")
        self.parser.add_argument("-o", "--outfile", type=str,default='', 
                    help="Outputs the report into the file specified. Defaults to standard output.")
        
    def parse(self):
        return self.parser.parse_args()

        
class CrawlerArgsParser(object):
    """
    This class is a parser for the Selenium based web crawler module arguments.
    """
    leagues = ['Primer_League','Serie_A','La_Liga','Ligue1','Bundesliga']
    leagues_links = {'PL_2010':'http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/2458/Stages/4345/Show/England-Premier-League-2010-2011',\
                        'PL_2011':'http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/2935/Stages/5476/Show/England-Premier-League-2011-2012',\
                        'PL_2012':'http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/3389/Stages/6531/Show/England-Premier-League-2012-2013',\
                        'PL_2013':'http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/3853/Stages/7794/Show/England-Premier-League-2013-2014',\
                        'PL_2014':'http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/4311/Stages/9155/Show/England-Premier-League-2014-2015',\
                        'PL_2015':'http://www.whoscored.com/Regions/252/Tournaments/2/England-Premier-League',\
                        'SA_2010':'http://www.whoscored.com/Regions/108/Tournaments/5/Seasons/2626/Stages/4659/Show/Italy-Serie-A-2010-2011',\
                        'SA_2011':'http://www.whoscored.com/Regions/108/Tournaments/5/Seasons/3054/Stages/5667/Show/Italy-Serie-A-2011-2012',\
                        'SA_2012':'http://www.whoscored.com/Regions/108/Tournaments/5/Seasons/3512/Stages/6739/Show/Italy-Serie-A-2012-2013',\
                        'SA_2013':'http://www.whoscored.com/Regions/108/Tournaments/5/Seasons/3978/Stages/8019/Show/Italy-Serie-A-2013-2014',\
                        'SA_2014':'http://www.whoscored.com/Regions/108/Tournaments/5/Seasons/5441/Stages/11369/Show/Italy-Serie-A-2014-2015',\
                        'SA_2015':'http://www.whoscored.com/Regions/108/Tournaments/5/Italy-Serie-A',\
                        'LL_2010':'http://www.whoscored.com/Regions/206/Tournaments/4/Seasons/2596/Stages/4624/Show/Spain-La-Liga-2010-2011',\
                        'LL_2011':'http://www.whoscored.com/Regions/206/Tournaments/4/Seasons/3004/Stages/5577/Show/Spain-La-Liga-2011-2012',\
                        'LL_2012':'http://www.whoscored.com/Regions/206/Tournaments/4/Seasons/3470/Stages/6652/Show/Spain-La-Liga-2012-2013',\
                        'LL_2013':'http://www.whoscored.com/Regions/206/Tournaments/4/Seasons/3922/Stages/7920/Show/Spain-La-Liga-2013-2014',\
                        'LL_2014':'http://www.whoscored.com/Regions/206/Tournaments/4/Seasons/5435/Stages/11363/Show/Spain-La-Liga-2014-2015',\
                        'LL_2015':'http://www.whoscored.com/Regions/206/Tournaments/4/Spain-La-Liga',\
                        'BL_2010':'http://www.whoscored.com/Regions/81/Tournaments/3/Seasons/2520/Stages/4448/Show/Germany-Bundesliga-2010-2011',\
                        'BL_2011':'http://www.whoscored.com/Regions/81/Tournaments/3/Seasons/2949/Stages/5492/Show/Germany-Bundesliga-2011-2012',\
                        'BL_2012':'http://www.whoscored.com/Regions/81/Tournaments/3/Seasons/3424/Stages/6576/Show/Germany-Bundesliga-2012-2013',\
                        'BL_2013':'http://www.whoscored.com/Regions/81/Tournaments/3/Seasons/3863/Stages/7806/Show/Germany-Bundesliga-2013-2014',\
                        'BL_2014':'http://www.whoscored.com/Regions/81/Tournaments/3/Seasons/4336/Stages/9192/Show/Germany-Bundesliga-2014-2015',\
                        'BL_2015':'http://www.whoscored.com/Regions/81/Tournaments/3/Germany-Bundesliga',\
                        'L1_2010':'http://www.whoscored.com/Regions/74/Tournaments/22/Seasons/2417/Stages/4273/Show/France-Ligue-1-2010-2011',\
                        'L1_2011':'http://www.whoscored.com/Regions/74/Tournaments/22/Seasons/2920/Stages/5451/Show/France-Ligue-1-2011-2012',\
                        'L1_2012':'http://www.whoscored.com/Regions/74/Tournaments/22/Seasons/3356/Stages/6476/Show/France-Ligue-1-2012-2013',\
                        'L1_2013':'http://www.whoscored.com/Regions/74/Tournaments/22/Seasons/3836/Stages/7771/Show/France-Ligue-1-2013-2014',\
                        'L1_2014':'http://www.whoscored.com/Regions/74/Tournaments/22/Seasons/4279/Stages/9105/Show/France-Ligue-1-2014-2015',\
                        'L1_2015':'http://www.whoscored.com/Regions/74/Tournaments/22/France-Ligue-1',}


    def __init__(self):
        self.multi = False
        self.range_kwargs= []
        self.parser = ArgumentParser(description='Crawl whoscored.com for the specified league and years.')
        self.parser.add_argument('league', metavar='League', type=str,
                           help='A league to parse. The leagues are: '+', '.join(self.leagues),
                           choices=self.leagues,nargs='?')
        self.parser.add_argument('year', metavar='Years',type=str, nargs='?',
                            help='A year to parse. Valid years are: '+', '.join([str(i) for i in range(MIN_YEAR,MAX_YEAR)]+[' or any range of them, separated by -.']),
                            default=str(max(range(2010,2015))),\
                            choices=self._default_ranges())
        self.parser.add_argument('-u','--update',action='store_true',help='Crawel current year on all leagues for update')
     
     
    def _default_ranges(self):
        """
        This function defines the defult ranges for the crawler.
        
        Using our constants MIN_YEAR and MAX_YEAR.
        """
        return [str(i) for i in range(MIN_YEAR,MAX_YEAR)]+['-'.join([str(i),str(j)]) for i in range(MIN_YEAR,MAX_YEAR) for j in range(MIN_YEAR,MAX_YEAR) if i<j]

    def parse(self):
        args = self.parser.parse_args()
        self.kwargs={}
        if not args.update:
            self.LEAGUE_NAME = vars(args)['league']
            self.update = False
            if '-' in vars(args)['year']:
                self.multi = True
                start , end = tuple(vars(args)['year'].split('-'))
                for year in range(int(start),int(end)+1):
                    self.kwargs['league'] = self.leagues_links[self._hash_league_names_and_years(self.LEAGUE_NAME,year)]
                    self.kwargs['year'] = int(year)
                    self.range_kwargs.append(dict(self.kwargs))
            else:
                self.kwargs['year'] = int(vars(args)['year'])
                self.kwargs['league'] = self.leagues_links[self._hash_league_names_and_years(self.LEAGUE_NAME,self.kwargs['year'])]
        else:
            self.update = True
            self.update_kwargs = []
            for league in self.leagues:
                self.kwargs['league'] = self.leagues_links[self._hash_league_names_and_years(league,MAX_YEAR)]
                self.kwargs['year'] = MAX_YEAR
                self.kwargs['r_league'] = league
                self.update_kwargs.append(dict(self.kwargs))
        
    def _hash_league_names_and_years(self,league,year):
        """
        This function is used as a hash function f:(long_league_name) --> short_league_name
        
        For example - f(Primer_League) = PL.
        """
        abv_league = ''
        if league == 'Primer_League':
            abv_league = 'PL'
        elif league == 'La_Liga':
            abv_league = 'LL'
        elif league == 'Bundesliga':
            abv_league = 'BL'
        elif league == 'Ligue1':
            abv_league = 'L1'
        elif league == 'Serie_A':
            abv_league = 'SA'
        return '_'.join([abv_league,str(year)])
    