'''
Created on Nov 27, 2015

@author: Ory Jonay
'''
import glob
from os import path
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from unidecode import unidecode


class WhoScoredCrawler(object):
    '''
    WhoScoredCrawler - A web crawler and data mining tool for the whoscored.com site.
    
    Requires Google Chrome and ChromeDriver in order to work. 
    '''


    def __init__(self, league,year,link):
        '''
        Constructor.
        
        @keyword league: The specific league to collect the data from.
        @keyword year: The specific year to collect the data from.
        @keyword link: The link corresponding to the league and year.
        '''
        self.league = league
        self.year = year
        self.league_link = link
        self.chrome_oprtions = webdriver.ChromeOptions()
        self.chrome_oprtions.add_extension('AdBlock_v2.36.2.crx')
        self.driver = webdriver.Chrome(chrome_options=self.chrome_oprtions)
        self.driver.implicitly_wait(10)
        self._create_backup()
    
    def _create_backup(self):
        '''
        Helper function to create backup folders and files for the data mined.
        '''
        self._bkup_folder = '-'.join([self.league,str(self.year)])
        self._bkup_fixtures_links = path.join(self._bkup_folder,'fixtures.pckl')  
        if not path.exists(self._bkup_folder):
            os.mkdir(self._bkup_folder)
            
    def get_played_months(self):
        '''
        Function to get the played months for the league.
        '''
        browser = self.driver
        config_button = browser.find_element_by_id('date-config-toggle-button')
        config_button.click()
    #     months_menu = browser.find_element_by_id('date-config')
        years = browser.find_elements_by_xpath('//*[@id="date-config"]/div[1]/div/table/tbody/tr/td[1]/div/table/tbody/tr/td')
        rows = browser.find_elements_by_xpath('//*[@id="date-config"]/div[1]/div/table/tbody/tr/td[2]/div/table/tbody/tr')
        months = []
        for row in reversed(rows):
            months += [str(e.text) for e in reversed(row.find_elements_by_class_name('selectable'))]
        years[0].click()
        rows = browser.find_elements_by_xpath('//*[@id="date-config"]/div[1]/div/table/tbody/tr/td[2]/div/table/tbody/tr')
        for row in reversed(rows):
            months += [str(e.text) for e in reversed(row.find_elements_by_class_name('selectable'))]
        config_button.click()
        return months  
    
    def crawl(self):
        '''
        The main function - crawl the league and mine some data.
        
        
        '''
        self.driver.get(self.league_link)
        self.team_names = set([unidecode(thr.text) for thr in \
                               self.driver.find_element_by_class_name("stat-table").find_elements_by_class_name("team-link")])
        self.driver.find_element(By.XPATH, '//*[@id="sub-navigation"]').find_element(By.PARTIAL_LINK_TEXT, 'Fixtures').click()
        self.played_months = self.get_played_months()
        try:
            self.start_month = self.find_start_month()
        except ValueError as e:
            if e.message == 'Finished':
                return
        
        
    def find_start_month(self):
        if path.exists(self._bkup_fixtures_links):
            pckl_files = glob.glob('%s/*.pckl'%self._bkup_folder)
            assert len(pckl_files) <= 2, 'Too many files, should only be fixtures and games for up to the last month'
            if len(pckl_files) == 1: return self.played_months[-1] 
            month = pckl_files[0].split('/')[1].split('.')[0] if pckl_files[0].split('/')[1].split('.')[0] in self.played_months else pckl_files[1].split('/')[1].split('.')[0]
            if month != self.played_months[0]:
                return self.played_months[self.played_months.index(month)-1]
            else:
                raise ValueError('Finished') 
        return self.played_months[-1]