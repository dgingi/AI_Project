#!/usr/bin/python
"""
Selenium Crawler module.

"""
 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait , Select
from selenium.webdriver.support import expected_conditions as EC
import time

from itertools import izip
from pickle import dump,load
from sys import argv
from unidecode import unidecode
from __builtin__ import str
import sys
from os import path, mkdir,remove
import utils
from utils import PrintException, DBHandler

def start_crawl(league,year,start_month='Aug'):
    """
    Start crawling and collecting data. 
    
    TODO: more generic crawling
    """
    chop = webdriver.ChromeOptions()
    chop.add_extension('AdBlock_v2.36.2.crx')
    finished = False
    while(not(finished)):
        browser = webdriver.Chrome(chrome_options = chop) # Get local session of Chrome
        browser.implicitly_wait(10)
        browser.get(league)
        select = Select(browser.find_element_by_id("seasons"))
        select.select_by_visible_text(str(year)+'/'+str(year+1))
        try:
            parse_league(browser,year,start_month)
        except Exception, e:
            browser.quit()
            PrintException()
            if e.args[0]=='Fin':
                return
            time.sleep(10)
            start_month = e.args[0]
            
def parse_league(browser,year,start_month):
    """
    Start parsing the league table.
    
    """
    def get_prev_month(month,all_months):
        if month=='Aug':
            return 'Aug'
        for i in range(len(all_months)-1):
            if all_months[i+1]==month:
                return all_months[i]
    
    def get_curr_fix(all_teams_dict,team_name):
        for key in all_teams_dict[team_name]:
            if not(all_teams_dict[team_name][key]):
                return key
            
    def get_fixtures(browser,file_pref,months):
        disp_last_week = browser.find_element_by_id("date-controller")
        curr_last_week = disp_last_week.find_elements_by_tag_name('a')[1]
        curr_last_week.click()
        last_played_month = browser.find_element_by_class_name('months')
        last_month = last_played_month.find_element_by_xpath('./tbody/tr/td[@class=" selected selectable"]')    
        if last_month.text == 'Jun':
            months+=['Jun']
        curr_last_week.click()
         
        fixtures_elm = browser.find_element_by_link_text("Fixtures")
        fixtures_elm.click()
        WebDriverWait(browser,10).until(EC.text_to_be_present_in_element((By.TAG_NAME,"h2"),'Fixture'))
        disp_month = browser.find_element_by_id('date-controller')
        prev_month = disp_month.find_elements_by_tag_name('a')[0]
        games_by_month = {i:None for i in months}
    
        for i in reversed(months):
            WebDriverWait(browser,30).until(EC.text_to_be_present_in_element((By.CLASS_NAME,"rowgroupheader"),i))
            all_res = browser.find_elements_by_xpath('//div[@id="tournament-fixture-wrapper"]/table/tbody/tr[@class!="rowgroupheader"]')
    
            games_by_month[i] = [{'link':unidecode(res.find_element_by_xpath('./td/a[@class="result-1 rc"]').get_attribute("href")),
                                  'home':unidecode(res.find_elements_by_xpath('./td[@data-id]/a')[0].text),
                                  'result':unidecode(res.find_element_by_xpath('./td/a[@class="result-1 rc"]').text),
                                  'away':unidecode(res.find_elements_by_xpath('./td[@data-id]/a')[1].text)} for res in all_res]
            prev_month.click()
        
        with open(file_pref+"/"+file_pref+"-fixtures.pckl",'w') as output:
            dump(games_by_month,output)
        return games_by_month      
    
    table = browser.find_element_by_class_name("stat-table")
    list_tlinks = table.find_elements_by_class_name("team-link")
    all_teams_names = set([unidecode(thr.text) for thr in list_tlinks])
    months = ['Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May']
    
    seq = (argv[1],str(year))
    file_pref = '-'.join(seq)
    if not path.exists(file_pref):
        mkdir(file_pref)
    
    if start_month != 'Aug' :
        with open(file_pref+"/"+file_pref+"-"+get_prev_month(start_month,months)+".pckl",'r') as res:
            all_teams_dict = load(res)
            all_teams_curr_fix = {name:get_curr_fix(all_teams_dict,name) for name in all_teams_names}
            with open(file_pref+"/"+file_pref+"-fixtures.pckl",'r') as output:
                games_by_month = load(output) 
    else:
        all_teams_dict = {name:{i:{} for i in range(1,2*len(all_teams_names)-1)} for name in all_teams_names}
        all_teams_curr_fix = {name:1 for name in all_teams_names}
        if not path.exists(file_pref+"/"+file_pref+"-fixtures.pckl"):
            games_by_month = get_fixtures(browser, file_pref, months)
        else:
            with open(file_pref+"/"+file_pref+"-fixtures.pckl",'r') as output:
                games_by_month = load(output)
        
    flag_of_start_month = False
    for month in months:
        if month == start_month:
            flag_of_start_month = True
        if not(flag_of_start_month):
            continue
        for game in games_by_month[month]:
            try:
                parse_game(browser,game,all_teams_dict,all_teams_curr_fix)
            except Exception, e:
                PrintException()
                raise Exception(month)  
        else: #saving each month separately
            with open(file_pref+"/"+file_pref+"-"+month+".pckl",'w') as output:
                dump(all_teams_dict, output)
            if month!='Aug':
                remove(file_pref+"/"+file_pref+"-"+get_prev_month(month,months)+".pckl")
    DBHandler(LEAGUE_NAME,str(year)).insert_to_db(all_teams_dict)
    raise Exception('Fin')
    

def parse_game(browser,game,all_teams_dict,all_teams_curr_fix):
    """
    Start parsing a single game (Home VS Away).

    """
    try:
        browser.get(game['link'])
        WebDriverWait(browser,15).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="sub-sub-navigation"]/ul/li[2]/a')))
        players_stats_link = browser.find_element_by_xpath('//*[@id="sub-sub-navigation"]/ul/li[2]/a').get_attribute("href")
        browser.get(players_stats_link)
         
        #time.sleep(10)
        
        home_team_name = game['home']
        away_team_name = game['away']
        
        stats = browser.find_element_by_id('match-report-team-statistics')
        raw_poss = stats.find_elements_by_xpath('./div[2]/div[2]/span/span[@class="stat-value"]')
        poss = [unidecode(r.text) for r in raw_poss]
        
        
        def create_dicts_from_table(table):
            players = table.find_elements_by_class_name("player-link")
            players_names = set([get_player_name(unidecode(player.text).split(' ')) for player in players])
            players_dict = {name:{"Summary":{},"Offensive":{},"Defensive":{},"Passing":{}} for name in players_names}
            return players_dict
        
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
        
        home_table = browser.find_element_by_id("live-player-home-stats")
        home_players_dict = create_dicts_from_table(home_table) 
        
        away_table = browser.find_element_by_id("live-player-away-stats")
        away_players_dict = create_dicts_from_table(away_table)
        
         
        parse_team(browser,"home",home_players_dict)
        parse_team(browser,"away",away_players_dict)
        
        raw_result = unidecode(browser.find_element_by_class_name("result").text).split(' ')
        result = [raw_result[0]]+[raw_result[2]]
        
        
        def update_team(all_teams_dict,all_teams_curr_fix,team_name,players_dict,HA,result,curr_poss,vs):
            team_curr_fix = all_teams_curr_fix[team_name]
            all_teams_dict[team_name][team_curr_fix]["Players"]=players_dict
            all_teams_dict[team_name][team_curr_fix]["HA"]=HA
            all_teams_dict[team_name][team_curr_fix]["Result"]=(result[0],result[1])
            all_teams_dict[team_name][team_curr_fix]["Tag"]=parse_result(result, HA)
            all_teams_dict[team_name][team_curr_fix]["Possession"]=curr_poss
            all_teams_dict[team_name][team_curr_fix]["VS"]=vs
            all_teams_curr_fix[team_name]+=1
            
            
        update_team(all_teams_dict, all_teams_curr_fix, home_team_name, home_players_dict, "home", result, poss[0],away_team_name)
        update_team(all_teams_dict, all_teams_curr_fix, away_team_name, away_players_dict, "away", result, poss[1],home_team_name)
    except:
        PrintException()
        raise  
    
def parse_team(browser,curr_team,all_players_dict):
    try:
        opt_tabels = browser.find_element_by_id("live-player-"+curr_team+"-options")
        linked_tabels = opt_tabels.find_elements_by_xpath(".//a")
        for link_table in linked_tabels:
            link_table.click()
            findStr = "live-player-"+curr_team+"-"+link_table.text.lower()
            WebDriverWait(browser,30).until(EC.visibility_of_element_located((By.ID,findStr)))
            rel_table = browser.find_element_by_id(findStr)
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
                player_name = get_player_name(unidecode(player_link.text).split(' '))
                if link_table.text.lower()=="summary":
                    player_pos = unidecode(all_player_info.find_elements_by_xpath('./span[@class="player-meta-data"]')[1].text).split(' ')[1]
                    goals = []
                    player_data_line += [player_pos]
                    try:
                        browser.implicitly_wait(1)
                        events = key_events.find_elements_by_xpath('./span/span')
                        goals = [event for event in events if (event.get_attribute("data-type")=="16") and not(event.get_attribute("data-event-satisfier-goalown"))]
                    finally:
                        browser.implicitly_wait(10)
                        player_data_line += [float(len(goals))]
                all_players_dict[player_name][link_table.text] = {h:d for h,d in izip(team_header,player_data_line)}
    except:
        PrintException()
        raise            
        
            
def get_player_name(str_list):
    str=""
    for i in range(len(str_list)):
        if 'A' <= str_list[i][0] <= 'Z':
            if i>0:
                str+=" "
            str+=str_list[i]
    return str


LEAGUE_NAME = ''

if __name__ == '__main__':
    import argparse
    leagues_links = {'Primer_League':"http://www.whoscored.com/Regions/252/Tournaments/2/England-Premier-League",
                     'Serie_A':"http://www.whoscored.com/Regions/108/Tournaments/5/Italy-Serie-A",
                     'La_Liga':"http://www.whoscored.com/Regions/206/Tournaments/4/Spain-La-Liga",
                     'Bundesliga':"http://www.whoscored.com/Regions/81/Tournaments/3/Germany-Bundesliga",
                     'Ligue_1':"http://www.whoscored.com/Regions/74/Tournaments/22/France-Ligue-1"
                     }
    parser = argparse.ArgumentParser(description='Crawel whoscored.com for the specified league and year.')
    parser.add_argument('league', metavar='League', type=str,
                       help='A league to parse. The leagues are: '+', '.join(leagues_links.keys()),choices=leagues_links.keys())
    parser.add_argument('year', metavar='Year',type=str, nargs='?',
                        help='A year to parse. Valid years: '+', '.join([str(i) for i in range(2010,2015)]),
                        default=str(max(range(2010,2015))),\
                        choices=['-'.join([str(i),str(j)]) for i in range(2010,2015) for j in range(2010,2015) if i<j])
    
    global LEAGUE_NAME
    args = parser.parse_args()
    kwargs={}
    LEAGUE_NAME = vars(args)['league']
    kwargs['league'] = leagues_links[vars(args)['league']]
    if '-' in vars(args)['year']:
        start , end = tuple(vars(args)['year'].split('-'))
        for year in range(int(start),int(end)+1):
            kwargs['year'] = int(year)
            start_crawl(**kwargs)
    else:
        kwargs['year'] = int(year)
        start_crawl(**kwargs)
    
    
