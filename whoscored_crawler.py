'''
Created on Nov 27, 2015

@author: Ory Jonay
'''
import glob
from itertools import izip
from os import path
import os
import pickle, logging
from progress.bar import ChargingBar
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from time import sleep
from unidecode import unidecode
from multiprocessing import Pool, cpu_count
from datetime import datetime

from data.dbhandler import DBHandler
from utils.argumet_parsers import CrawlerArgsParser
from utils.decorators import retry


args_parser = CrawlerArgsParser()


class WhoScoredCrawler(object):
    '''
    WhoScoredCrawler - A web crawler and data mining tool for the whoscored.com site.
    
    Requires Google Chrome and ChromeDriver in order to work. 
    '''


    def __init__(self, league,year,link):
        '''
        Constructor for the WhoScoredCrawler class.
        
        @keyword league: The specific league to collect the data from.
        @keyword year: The specific year to collect the data from.
        @keyword link: The link corresponding to the league and year.
        '''
        logging.basicConfig(filename=path.join('logs','%s%s.log'%(league,year)),format='%(levelname)s: %(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M %p',level=logging.INFO)
        self.league = league
        self.year = year
        self.league_link = link
        self.chrome_oprtions = webdriver.ChromeOptions()
        self.chrome_oprtions.add_extension('1.4.0_0.crx')
        self.driver = webdriver.Chrome(chrome_options=self.chrome_oprtions)
        self.driver.implicitly_wait(30)
        self._create_backup()
        if not path.exists('logs'):
            os.mkdir('logs')
        logging.info('Finished crawler initialization')
        
    def restart_driver(self):
        '''
        Restarts the driver.
        
        In case of the driver getting laggy and it needs a restart, quits, sleeps for 10 seconds and reopens the browser.
        '''
        self.driver.quit()
        sleep(10)
        self.driver = webdriver.Chrome(chrome_options=self.chrome_oprtions)
        self.driver.implicitly_wait(30)
        
    def _create_backup(self):
        '''
        Helper function to create backup folders and files for the data mined.
        '''
        logging.info('Creating backup folders')
        self._bkup_folder = path.join('backup','-'.join([self.league,str(self.year)]))
        self._bkup_fixtures_links = path.join(self._bkup_folder,'fixtures.pckl')  
        if not path.exists(self._bkup_folder):
            os.mkdir(self._bkup_folder)
        logging.info('Finished creating backup folders')
            
    def get_played_months(self):
        '''
        Function to get the played months for the league.
        '''
        logging.info('Finding played months')
        self.driver = self.driver
        config_button = self.driver.find_element_by_id('date-config-toggle-button')
        config_button.click()
    #     months_menu = self.driver.find_element_by_id('date-config')
        years = self.driver.find_elements_by_xpath('//*[@id="date-config"]/div[1]/div/table/tbody/tr/td[1]/div/table/tbody/tr/td')
        rows = self.driver.find_elements_by_xpath('//*[@id="date-config"]/div[1]/div/table/tbody/tr/td[2]/div/table/tbody/tr')
        months = []
        for row in reversed(rows):
            months += [str(e.text) for e in reversed(row.find_elements_by_class_name('selectable'))]
        years[0].click()
        rows = self.driver.find_elements_by_xpath('//*[@id="date-config"]/div[1]/div/table/tbody/tr/td[2]/div/table/tbody/tr')
        for row in reversed(rows):
            months += [str(e.text) for e in reversed(row.find_elements_by_class_name('selectable'))]
        config_button.click()
        logging.info('Finished getting played months')
        return months  
    
    def crawl(self,current=False):
        '''
        The main function - crawl the league and mine some data.
        
        
        '''
        logging.info('Starting crawl')
        self.driver.get(self.league_link)
        self.team_names = set([unidecode(thr.text) for thr in \
                               self.driver.find_element_by_class_name("stat-table").find_elements_by_class_name("team-link")])
        self.driver.find_element(By.XPATH, '//*[@id="sub-navigation"]').find_element(By.PARTIAL_LINK_TEXT, 'Fixtures').click()
        self.played_months = self.get_played_months()    
        self.load_previous_data(current)
        prog_bar = ChargingBar('Progress of %s crawling:'%' '.join([self.league,str(self.year)]),max=sum([len(self.fixtures[month]) for month in self.played_months[self.played_months.index(self.start_month)::-1]]))
        for month in self.played_months[self.played_months.index(self.start_month)::-1]:
            for game in self.fixtures[month]:
                logging.info('Starting to parse game')
                if game: 
                    self.parse_game(game)
                prog_bar.next()
                logging.info('Finished game, moving to the next one')
            else:
                logging.info('Finished month, saving to disk')
                self.save_month(month)
                if current:
                    pass
#                     DBHandler(args_parser.LEAGUE_NAME).update_db(self.all_teams_dict,str(self.year))
        else: #we're done - we can save to the DB now
#             if not current:
            DBHandler(args_parser.LEAGUE_NAME).insert_to_db(self.all_teams_dict,str(self.year))
        self.driver.quit()
        prog_bar.finish()
        
    @retry(True)        
    def parse_game(self,game):
        '''
        Parses a game and updates the dictionaries all_team_dict and all_teams_curr_fix
        '''
        self.driver.get(game['link'])
        WebDriverWait(self.driver,15).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="sub-sub-navigation"]/ul/li[2]/a')))
        players_stats_link = self.driver.find_element_by_xpath('//*[@id="sub-sub-navigation"]/ul/li[2]/a').get_attribute("href")
        self.driver.get(players_stats_link)
        
        home_team_name = game['home']
        away_team_name = game['away']
        
        stats = self.driver.find_element_by_id('match-report-team-statistics')
        raw_poss = stats.find_elements_by_xpath('./div[2]/div[2]/span/span[@class="stat-value"]')
        poss = [unidecode(r.text) for r in raw_poss]
        
        def create_dicts_from_table(table):
            players = table.find_elements_by_class_name("player-link")
            players_names = set([self.get_player_name(unidecode(player.text).split(' ')) for player in players])
            players_dict = {name:{"Summary":{},"Offensive":{},"Defensive":{},"Passing":{}} for name in players_names}
            return players_dict
        
        self.players_dict = {}
        self.players_dict['home'] = create_dicts_from_table(self.driver.find_element_by_id("live-player-home-stats"))
        self.players_dict['away'] = create_dicts_from_table(self.driver.find_element_by_id("live-player-away-stats"))
        
        self.parse_teams()
        
        raw_result = unidecode(self.driver.find_element_by_class_name("result").text).split(' ')
        result = [raw_result[0]]+[raw_result[2]]
        
        def update_team(team_name,HA,curr_poss,vs):
            def parse_result(result_list,curr_team):
                winner = "tie"
                if result_list[0] > result_list[1]:
                    winner = "home"
                elif result_list[0] < result_list[1]:
                    winner = "away"
                if winner == curr_team:
                    return 1
                elif winner != "tie":
                    return -1
                else:
                    return 0
            team_curr_fix = self.all_teams_curr_fix[team_name]
            self.all_teams_dict[team_name][team_curr_fix]["Players"]=self.players_dict[HA]
            self.all_teams_dict[team_name][team_curr_fix]["HA"]=HA
            self.all_teams_dict[team_name][team_curr_fix]["Result"]=(result[0],result[1])
            self.all_teams_dict[team_name][team_curr_fix]["Tag"]=parse_result(result, HA)
            self.all_teams_dict[team_name][team_curr_fix]["Possession"]=curr_poss
            self.all_teams_dict[team_name][team_curr_fix]["VS"]=vs
            self.all_teams_curr_fix[team_name]+=1
            
        update_team(home_team_name, "home", poss[0],away_team_name)
        update_team(away_team_name, "away", poss[1],home_team_name)
 
    def get_player_name(self,str_list):
        _str=""
        for i in range(len(str_list)):
            if 'A' <= str_list[i][0] <= 'Z':
                if i>0:
                    _str+=" "
                _str+=str_list[i]
        return _str
        
    def parse_teams(self):
        '''
        Parses the teams players tables - 8 tables in total.
        '''
        for dict in self.players_dict:
            opt_tabels = self.driver.find_element_by_id("live-player-"+dict+"-options")
            linked_tabels = opt_tabels.find_elements_by_xpath(".//a")
            for link_table in linked_tabels:
                link_table.click()
                findStr = "live-player-"+dict+"-"+link_table.text.lower()
                WebDriverWait(self.driver,30).until(EC.visibility_of_element_located((By.ID,findStr)))
                rel_table = self.driver.find_element_by_id(findStr)
                table = rel_table.find_element_by_id("top-player-stats-summary-grid")
                team_header = [unidecode(h.text) for h in table.find_element_by_tag_name("thead").find_elements_by_tag_name("th")[3:-2]]
                if link_table.text.lower()=="summary":
                    team_header += ["Position"]
                    team_header += ["Goals"]
                team_body = table.find_element_by_tag_name("tbody")
                team_lines = team_body.find_elements_by_tag_name("tr")
                for line in team_lines:
                    player_data_line = [float(l.text) for l in line.find_elements_by_tag_name("td")[3:-2]]
                    key_events = line.find_elements_by_tag_name("td")[-1]
                    player_link = line.find_element_by_class_name("player-link")    
                    all_player_info = line.find_element_by_class_name('pn')
                    player_name = self.get_player_name(unidecode(player_link.text).split(' '))
                    if link_table.text.lower()=="summary":
                        player_pos = unidecode(all_player_info.find_elements_by_xpath('./span[@class="player-meta-data"]')[1].text).split(' ')[1]
                        goals = []
                        player_data_line += [player_pos]
                        try:
                            self.driver.implicitly_wait(1)
                            events = key_events.find_elements_by_xpath('./span/span')
                            goals = [event for event in events if (event.get_attribute("data-type")=="16") and not(event.get_attribute("data-event-satisfier-goalown"))]
                        finally:
                            self.driver.implicitly_wait(30)
                            player_data_line += [float(len(goals))]
                    self.players_dict[dict][player_name][link_table.text] = {h:d for h,d in izip(team_header,player_data_line)}
       
    def load_previous_data(self,current=False):
        '''
        Loads the data collected in previous run in order to resume crawling from that point.
        '''
        logging.info('Loading data from previous runs')
        if not current:
            if not path.exists(self._bkup_fixtures_links):
                self.get_fixtures()
            else:
                with open(self._bkup_fixtures_links,'rb') as _bkup_f:
                    self.fixtures = pickle.load(_bkup_f)
        else:
            self.get_fixtures(current)
        try:
            self.start_month = self.find_start_month(current)
        except ValueError as e:
            if e.message == 'Finished':
                with open(path.join(self._bkup_folder,self.last_save_month+'.pckl'),'rb') as _bkup_f:
                    self.all_teams_dict = pickle.load(_bkup_f)
                return
        if self.start_month != self.played_months[-1]:
            with open(path.join(self._bkup_folder,self.last_save_month+'.pckl'),'rb') as _bkup_f:
                self.all_teams_dict = pickle.load(_bkup_f)
            if not current:
                with open(path.join(self._bkup_folder,self.last_save_month+'-nxtFix.pckl'),'rb') as _bkup_f:
                    self.all_teams_curr_fix = pickle.load(_bkup_f)
            else:
                if self.last_save_month != datetime.now().strftime('%b'):
                    with open(path.join(self._bkup_folder,self.last_save_month+'-nxtFix.pckl'),'rb') as _bkup_f:
                        self.all_teams_curr_fix = pickle.load(_bkup_f)
                else:
                    with open(path.join(self._bkup_folder,self.played_months[self.played_months.index(self.last_save_month)+1]+'-nxtFix.pckl'),'rb') as _bkup_f:
                        self.all_teams_curr_fix = pickle.load(_bkup_f)
        else:
            self.all_teams_dict = {name:{i:{} for i in range(1,2*len(self.team_names)-1)} for name in self.team_names}
            self.all_teams_curr_fix = {name:1 for name in self.team_names}
        logging.info('Finished loading previous data')
            
    def _get_curr_fix(self,team_name):
        for key in self.all_teams_dict[team_name]:
            if not(self.all_teams_dict[team_name][key]):
                return key
    
    def get_fixtures(self,current=False):
        '''
        Parse the fixtures page for the games links and save them.
        '''
        logging.info('Getting fixtures')
        self.played_months = self.played_months[:self.played_months.index(datetime.now().strftime('%b'))-1:-1][::-1] if current else self.played_months 
        self.fixtures = {month:None for month in self.played_months}
        prev_month = self.driver.find_element_by_xpath('//*[@id="date-controller"]/a[1]')
        xpath_query = '//div[@id="tournament-fixture-wrapper"]/table/tbody/tr[@class!="rowgroupheader"]' if not current else '//div[@id="tournament-fixture-wrapper"]/table/tbody/tr[@class!="rowgroupheader"]//a[@class="result-1 rc"]/../..'
        for month in self.played_months:
            WebDriverWait(self.driver,30).until(EC.text_to_be_present_in_element((By.CLASS_NAME,"rowgroupheader"),month))
            all_res = self.driver.find_elements_by_xpath(xpath_query)
            self.fixtures[month] = [self.parse_fixture(res) for res in all_res]
            prev_month.click()
        self.save_fixtures()
        logging.info('Finished fixtures')
        
    def save_fixtures(self):
        with open(self._bkup_fixtures_links,'wb') as fixutres_bkup:
            pickle.dump(self.fixtures, fixutres_bkup)
    
    def save_month(self,month):
        with open(path.join(self._bkup_folder,month+'.pckl'),'wb') as month_bkup:
            pickle.dump(self.all_teams_dict,month_bkup)
        nxt_fix_dict = {name:self._get_curr_fix(name) for name in self.team_names}
        with open(path.join(self._bkup_folder,month+'-nxtFix.pckl'),'wb') as month_nf_bkup:
            pickle.dump(nxt_fix_dict,month_nf_bkup)
        if month != self.played_months[-1]:
            os.remove(path.join(self._bkup_folder,self.played_months[self.played_months.index(month)+1]+'.pckl'))
            if os.path.exists(path.join(self._bkup_folder,self.played_months[self.played_months.index(month)+2]+'-nxtFix.pckl')):
                try:
                    os.remove(path.join(self._bkup_folder,self.played_months[self.played_months.index(month)+2]+'-nxtFix.pckl'))
                except Exception as e:
                    logging.info(str(e))
                    pass
            
    @retry()       
    def parse_fixture(self,fixture):
        '''
        Parses a fixture and return a dictionary containing all data concerning that fixture.
        '''
        link = unidecode(fixture.find_element_by_xpath('./td/a[@class="result-1 rc"]').get_attribute("href"))
        home = unidecode(fixture.find_elements_by_xpath('./td[@data-id]/a')[0].text)
        result = unidecode(fixture.find_element_by_xpath('./td/a[@class="result-1 rc"]').text)
        away = unidecode(fixture.find_elements_by_xpath('./td[@data-id]/a')[1].text)
        return {'link':link,'home':home,'result':result,'away':away}
                                  
    def find_start_month(self,current=False):
        '''
        Finding the start month for the current run of the crawler.
        '''
        
        logging.info('Finding the start month')
        if path.exists(self._bkup_fixtures_links):
            pckl_files = glob.glob('%s/*.pckl'%self._bkup_folder)
            assert len(pckl_files) <= 3, 'Too many files, should only be fixtures and games for up to the last month'
            if len(pckl_files) == 1: return self.played_months[-1] 
            #TODO fix to match 3 files
            def _get_last_save_month():
                for i in range(len(pckl_files)):
                    if pckl_files[i].split('/')[2].split('.')[0] in self.played_months:
                        return pckl_files[i].split('/')[2].split('.')[0]
            self.last_save_month = _get_last_save_month()
            if not current:
                if self.last_save_month != self.played_months[0]:
                    logging.info('Finished finding start month')
                    return self.played_months[self.played_months.index(self.last_save_month)-1]
                else:
                    logging.critical('Finished crawling')
                    raise ValueError('Finished')
            else:
                if self.last_save_month == datetime.now().strftime('%b'):    
                    return self.played_months[self.played_months.index(self.last_save_month)]
                else:
                    return self.played_months[self.played_months.index(self.last_save_month)-1]
        logging.info('Finished finding start month')
        return self.played_months[-1]
    
    
    
def start_crawl(kwargs):
    '''
    Start crawling, given the arguments
    '''
    year = kwargs['year']
    league = kwargs['league']
    if args_parser.update:
        args_parser.LEAGUE_NAME = kwargs['r_league']
    WhoScoredCrawler(args_parser.LEAGUE_NAME,year,league).crawl(args_parser.update)
    
    
if __name__ == '__main__':
    args_parser.parse()
    if not args_parser.update:
        if args_parser.multi:
            p = Pool(cpu_count())
            p.map(start_crawl, args_parser.range_kwargs)
        else:
            start_crawl(args_parser.kwargs)
    else:
#         for kwargs in args_parser.update_kwargs:
#             start_crawl(kwargs)
        p = Pool(cpu_count()/2)
        p.map(start_crawl, args_parser.update_kwargs)
