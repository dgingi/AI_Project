"""
Selenium Crawler module.

"""
 
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from itertools import izip
from pickle import dump,load

def start_crawl():
    """
    Start crawling and collecting data. 
    
    TODO: more generic crawling
    """
    browser = webdriver.Chrome() # Get local session of Chrome
    browser.get("http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/3853")
    browser.implicitly_wait(5)
    parse_league(browser)
    
    browser.close()
    
def parse_league(browser):
    """
    Start parsing the league table.
    
    """
    wait = WebDriverWait(browser,5)
    
    table = browser.find_element_by_class_name("stat-table")
    list_tlinks = table.find_elements_by_class_name("team-link")
    all_teams_names = set([thr.text for thr in list_tlinks])
    max_games = 2*len(all_teams_names)-1
    
    all_teams_dict = {name:{i:{} for i in range(1,max_games)} for name in all_teams_names}
    all_teams_curr_fix = {name:1 for name in all_teams_names}
    
    fixtures_elm = browser.find_element_by_link_text("Fixtures")
    fixtures_elm.click()
    try:
        wait.until(EC.text_to_be_present_in_element((By.TAG_NAME,"h2"),'Fixture'))
    finally:
        pass
    
    disp_month = browser.find_element_by_id('date-controller')
    prev_month = disp_month.find_elements_by_tag_name('a')[0]
    months = ['Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May']
    games_by_month = {i:None for i in months}

    for i in months[::-1]:
        try:
            wait.until(EC.text_to_be_present_in_element((By.CLASS_NAME,"rowgroupheader"),i))
            all_res = browser.find_elements_by_xpath('//a[@class="result-1 rc"]')
            games_by_month[i] = [(res.get_attribute("href"),res.text) for res in all_res]
            prev_month.click()
        finally: #TODO - catch the TimeOut exception and do something about it
            pass
    
    for month in months:
        for game in games_by_month[month]:
            parse_game(browser,game[0],all_teams_dict,all_teams_curr_fix)  
    
    with open("PL-13-14.pckl",'w') as output:
        dump(all_teams_dict, output)  
    
   
def parse_game(browser,link,all_teams_dict,all_teams_curr_fix):
    """
    Start parsing a single game (Home VS Away).

    """
    
    browser.get(link)
    
    players_stats_elm = browser.find_element_by_link_text("Player Statistics")
    players_stats_elm.click() #TODO - see if need to wait for page to load
    
    list_tlinks = browser.find_elements_by_class_name("team-link")
    home_team_name = list_tlinks[0].text
    away_team_name = list_tlinks[1].text
    
    home_table = browser.find_element_by_id("live-player-home-stats")
    home_players = home_table.find_elements_by_class_name("pn")
    home_players_names = set([player.text.split('\n')[0] for player in home_players])
    home_players_dict = {name:{"summary":{},"offensive":{},"defensive":{},"passing":{}} for name in home_players_names}
    
    away_table = browser.find_element_by_id("live-player-away-stats")
    away_players = away_table.find_elements_by_class_name("pn")
    away_players_names = set([player.text.split('\n')[0] for player in away_players])
    away_players_dict = {name:{"summary":{},"offensive":{},"defensive":{},"passing":{}} for name in away_players_names}
    
     
    parse_team(browser,"home",home_players_dict)
    parse_team(browser,"away",away_players_dict)
    
    all_teams_dict[home_team_name][all_teams_curr_fix[home_team_name]]=home_players_dict
    all_teams_curr_fix[home_team_name]+=1
    all_teams_dict[away_team_name][all_teams_curr_fix[away_team_name]]=away_players_dict
    all_teams_curr_fix[away_team_name]+=1
    
def parse_team(browser,curr_team,all_players_dict):
    opt_tabels = browser.find_elements_by_id("live-player-"+curr_team+"-options")[0]
    linked_tabels = opt_tabels.find_elements_by_xpath(".//a")
    for link_table in linked_tabels:
        link_table.click()
        rel_table = browser.find_elements_by_id("live-player-"+curr_team+"-"+link_table.text.lower())[0]
        table = rel_table.find_elements_by_id("top-player-stats-summary-grid")[0]
        team_header = [h.text for h in table.find_element_by_tag_name("thead").find_elements_by_tag_name("th")[7:]]
        team_body = table.find_element_by_tag_name("tbody")
        team_lines = team_body.find_elements_by_tag_name("tr")
        for line in team_lines:
            player_data_line = [l.text for l in line.find_elements_by_tag_name("td")[2:]]
            player_name = player_data_line[0].split('\n')[0]
            player_data_line = player_data_line[1:]
            all_players_dict[player_name][link_table.text.lower()] = {h:d for h,d in izip(team_header,player_data_line)}
    
        
            
if __name__ == '__main__':
    start_crawl() 