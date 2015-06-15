"""
Selenium Crawler module.

"""
 
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from itertools import izip
from pickle import dump,load
import utils
from _elementtree import Element

def start_crawl():
    """
    Start crawling and collecting data. 
    
    TODO: more generic crawling
    """
    browser = webdriver.Chrome() # Get local session of Chrome
    browser.implicitly_wait(10)
    browser.get("http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/3853")
    parse_league(browser)
    
    browser.close()
    
def parse_league(browser):
    """
    Start parsing the league table.
    
    """
    
    table = browser.find_element_by_class_name("stat-table")
    list_tlinks = table.find_elements_by_class_name("team-link")
    all_teams_names = set([thr.text for thr in list_tlinks])
    
    
    all_teams_dict = {name:{i:{} for i in range(1,2*len(all_teams_names)-1)} for name in all_teams_names}
    all_teams_curr_fix = {name:1 for name in all_teams_names}
    
    fixtures_elm = browser.find_element_by_link_text("Fixtures")
    fixtures_elm.click()
    WebDriverWait(browser,10).until(EC.text_to_be_present_in_element((By.TAG_NAME,"h2"),'Fixture'))
    disp_month = browser.find_element_by_id('date-controller')
    prev_month = disp_month.find_elements_by_tag_name('a')[0]
    months = ['Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May']
    games_by_month = {i:None for i in months}

    for i in months[::-1]:
        WebDriverWait(browser,10).until(EC.text_to_be_present_in_element((By.CLASS_NAME,"rowgroupheader"),i))
        all_res = browser.find_elements_by_xpath('//a[@class="result-1 rc"]')
        games_by_month[i] = [(res.get_attribute("href"),res.text) for res in all_res]
        prev_month.click()
            
    for month in months[:7]:
        for game in games_by_month[month]:
            parse_game(webdriver.Chrome(),game[0],all_teams_dict,all_teams_curr_fix)  
        else: #saving each month separately
            with open("PL-13-14-"+month+".pckl",'w') as output:
                dump(all_teams_dict, output)  
    
    

def parse_game(browser,link,all_teams_dict,all_teams_curr_fix):
    """
    Start parsing a single game (Home VS Away).

    """
    browser.get(link)
    browser.implicitly_wait(10)
    WebDriverWait(browser,15).until(EC.text_to_be_present_in_element((By.ID,"sub-sub-navigation"),"Player Statistics"))
    players_stats_elm = browser.find_element_by_link_text("Player Statistics")
    players_stats_elm.click()
     
    time.sleep(10)
    
    list_tlinks = browser.find_elements_by_class_name("team-link")
    home_team_name = list_tlinks[0].text
    away_team_name = list_tlinks[1].text
    
    
    def create_dicts_from_table(table):
        players = table.find_elements_by_class_name("player-link")
        players_names = set([player.text for player in players])
        players_dict = {name:{"summary":{},"offensive":{},"defensive":{},"passing":{}} for name in players_names}
        return players_dict
    
    
    home_table = browser.find_element_by_id("live-player-home-stats")
    home_players_dict = create_dicts_from_table(home_table) 
    
    away_table = browser.find_element_by_id("live-player-away-stats")
    away_players_dict = create_dicts_from_table(away_table)
    
     
    parse_team(browser,"home",home_players_dict)
    parse_team(browser,"away",away_players_dict)
    
    all_teams_dict[home_team_name][all_teams_curr_fix[home_team_name]]=home_players_dict
    all_teams_curr_fix[home_team_name]+=1
    all_teams_dict[away_team_name][all_teams_curr_fix[away_team_name]]=away_players_dict
    all_teams_curr_fix[away_team_name]+=1
    browser.close()
    
def parse_team(browser,curr_team,all_players_dict):
    opt_tabels = browser.find_element_by_id("live-player-"+curr_team+"-options")
    linked_tabels = opt_tabels.find_elements_by_xpath(".//a")
    for link_table in linked_tabels:
        link_table.click()
        findStr = "live-player-"+curr_team+"-"+link_table.text.lower()
        WebDriverWait(browser,10).until(EC.visibility_of_element_located((By.ID,findStr)))
        rel_table = browser.find_element_by_id(findStr)
        table = rel_table.find_element_by_id("top-player-stats-summary-grid")
        team_header = [h.text for h in table.find_element_by_tag_name("thead").find_elements_by_tag_name("th")[3:-2]]
        if link_table.text.lower()=="summary":
            team_header += ["Goals"]
        team_body = table.find_element_by_tag_name("tbody")
        team_lines = team_body.find_elements_by_tag_name("tr")
        for line in team_lines:
            player_data_line = [float(l.text) for l in line.find_elements_by_tag_name("td")[3:-2]]
            key_events = line.find_elements_by_tag_name("td")[-1]
            player_name = line.find_element_by_class_name("player-link").text
            if link_table.text.lower()=="summary":
                try:
                    browser.implicitly_wait(1)
                    events = key_events.find_elements_by_xpath('./span/span')
                    goals = [event for event in events if (event.get_attribute("data-type")=="16") and not(event.get_attribute("data-event-satisfier-goalown"))]
                finally:
                    goals = []
                    browser.implicitly_wait(10)
                    player_data_line += [float(len(goals))]
            all_players_dict[player_name][link_table.text.lower()] = {h:d for h,d in izip(team_header,player_data_line)}
    
        
            
if __name__ == '__main__':
    start_crawl() 