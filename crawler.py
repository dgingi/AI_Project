"""
Selenium Crawler module.

"""
 
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
import time
from itertools import izip

def start_crawl():
    """
    Start crawling and collecting data. 
    
    TODO: more generic crawling
    """
    browser = webdriver.Chrome() # Get local session of Chrome
    browser.get("http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/4311/Stages/9155/TeamStatistics/England-Premier-League-2014-2015") # Load page
    browser.implicitly_wait(10)
    parse_league(browser)
    
    browser.close()
    
def parse_league(browser):
    """
    Start parsing the league table.
    
    
    TODO: parse the other tables
    """
    league_data = []
    table = browser.find_elements_by_id("top-team-stats-summary-grid")[0]
    league_header = [h.text for h in table.find_element_by_tag_name("thead").find_elements_by_tag_name("th")]
    
    league_body = table.find_element_by_tag_name("tbody")
    league_lines = league_body.find_elements_by_tag_name("tr")
    for line in league_lines:
        team_data_line = [l.text for l in line.find_elements_by_tag_name("td")]
        league_data.append({h:d for h,d in izip(league_header,team_data_line) if h!="Discipline"})
    
    list_tlinks = browser.find_elements_by_class_name("team-link")
    list_thr = set([thr.get_attribute("href") for thr in list_tlinks])
    for link in list_thr:
        parse_team(browser,link,league_data)
    
    for d in league_data:
        print (d['Team'].encode('utf-8'),len(d['Players']))
    
def parse_team(browser,link,league_data):
    """
    Start parsing the team table.
    
    TODO: parse the other tables
    """
    browser.get(link)
    browser.implicitly_wait(10)
    
    team_name = browser.find_element_by_tag_name("h1").text
    table = browser.find_elements_by_id("top-player-stats-summary-grid")[0]
    team_header = [h.text for h in table.find_element_by_tag_name("thead").find_elements_by_tag_name("th")]
    players_data = []
    team_body = table.find_element_by_tag_name("tbody")
    team_lines = team_body.find_elements_by_tag_name("tr")
    for line in team_lines:
        player_data_line = [l.text for l in line.find_elements_by_tag_name("td")]
        player_dict = {h:d for h,d in izip(team_header,player_data_line) if h!=""}
        p_name = player_dict['Name'].split('\n')[0]
        age,pos = player_dict['Name'].split('\n')[1].split(' ')
        player_dict['Name'] = p_name
        player_dict['Position'] = pos
        player_dict['Age'] = age[:-1]
        players_data.append(player_dict)
    for team in league_data:
        if team['Team'] == team_name:
            team['Players'] = players_data
            
if __name__ == '__main__':
    start_crawl() 