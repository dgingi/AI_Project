'''
Created on Jun 29, 2016

Crawler for oddsportal.com

@author: Ory Jonay
'''

import os
import sys; sys.path.append('..')
from os.path import join
from selenium import webdriver
import time
from utils.constants import MIN_YEAR, MAX_YEAR, LEAGUES_ABV
from utils.decorators import retry

def _us2dec(odd):
    if int(odd) > 0:
        return int(odd)/100. + 1
    else:
        return 100./abs(int(odd)) + 1

class OddsPortalCrawler(object):
    
    def __init__(self,url):
        self.url = url
        self.chrome_oprtions = webdriver.ChromeOptions()
        self.chrome_oprtions.add_extension(os.path.abspath(join(os.path.dirname(__file__),'..','1.4.0_0.crx')))
        self.driver = webdriver.Chrome(os.path.abspath(join(os.path.dirname(__file__),'..','chromedriver')),
                                       chrome_options=self.chrome_oprtions)
        self.driver.implicitly_wait(30)
        self.driver.get(self.url)
    
    @retry()
    def crawl(self):
        last_page = self.driver.find_elements_by_xpath('//a[@x-page]')[-1].get_attribute("x-page")
        pages = [self.url+'#/page/%d/'%i for i in range(1,int(last_page)+1)]
        games = {}
        for page in pages:
            self.driver.get(page)
            time.sleep(2)
            table = self.driver.find_element_by_id('tournamentTable')
            rows = table.find_elements_by_xpath('//tr[@class="odd deactivate"]')
            for row in rows:
                game , odds = row.find_element_by_class_name('table-participant').text, row.find_elements_by_class_name('odds-nowrp')
                games[game] = [float(o.text) for o in odds]
        self.driver.quit()
        return games
    
if __name__ == '__main__':
    base_pl_url = 'http://www.oddsportal.com/soccer/england/premier-league-{0}-{1}/results/'
    base_l1_url = 'http://www.oddsportal.com/soccer/france/ligue-1-{0}-{1}/results/'
    base_bl_url = 'http://www.oddsportal.com/soccer/germany/bundesliga-{0}-{1}/results/'
    base_ll_url = 'http://www.oddsportal.com/soccer/spain/primera-division-{0}-{1}/results/'
    base_sa_url = 'http://www.oddsportal.com/soccer/italy/serie-a-{0}-{1}/results/'
    res = {abv:{} for abv in LEAGUES_ABV}
    for i in range(MIN_YEAR,MAX_YEAR):
        res['PL'][i] = OddsPortalCrawler(base_pl_url.format(i,i+1)).crawl()
        res['SA'][i] = OddsPortalCrawler(base_sa_url.format(i,i+1)).crawl()
        res['L1'][i] = OddsPortalCrawler(base_l1_url.format(i,i+1)).crawl()
        res['BL'][i] = OddsPortalCrawler(base_bl_url.format(i,i+1)).crawl()
        res['LL'][i] = OddsPortalCrawler(base_ll_url.format(i,i+1)).crawl()
    
    from pickle import dump
    with open(os.path.join(os.path.dirname(__file__),'odds.pckl'),'w') as save_file:
        dump(res,save_file)