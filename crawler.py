"""
Selenium Crawler module.

"""
 
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
import time
from itertools import izip
from pickle import dump,load

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

    
    list_tlinks = browser.find_elements_by_class_name("team-link")
    list_thr = set([thr.get_attribute("href") for thr in list_tlinks])
    
    all_teams_names = set([thr.text for thr in list_tlinks])
    all_teams_dict = {name:[{},{},{},{}] for name in all_teams_names}
    
    opt_tabels = browser.find_elements_by_id("stage-team-stats-options")[0]
    linked_tabels = opt_tabels.find_elements_by_xpath(".//a")[:-1] 
    
    iter=0 
    
    for link_table in linked_tabels:
        link_table.click()
        link_table.parent.implicitly_wait(10)
        rel_table = browser.find_elements_by_id("stage-team-stats-"+link_table.text.lower())[0]
        table = rel_table.find_elements_by_id("top-team-stats-summary-grid")[0]
        league_header = [h.text for h in table.find_element_by_tag_name("thead").find_elements_by_tag_name("th")[2:]]
        league_body = table.find_element_by_tag_name("tbody")
        league_lines = league_body.find_elements_by_tag_name("tr")
        for line in league_lines:
            team_data_line = [l.text for l in line.find_elements_by_tag_name("td")[1:]]
            team_name = team_data_line[0]
            team_data_line = team_data_line[1:]
            all_teams_dict[team_name][iter] = {h:d for h,d in izip(league_header,team_data_line) if h!="Discipline"}
        iter+=1
           
    for link in list_thr:
        parse_team(browser,link,all_teams_dict)
    
    with open("output.pckl",'w') as output:
        dump(all_teams_dict, output)
    #for d in all_teams_dict:
        #print (d.encode('utf-8'),all_teams_dict[d])
    
    
    
def parse_team(browser,link,all_teams_dict):
    """
    Start parsing the team table.
    
    TODO: parse the other tables
    """
    
    browser.get(link)
    browser.implicitly_wait(10)
    
    #list_plinks = browser.find_elements_by_class_name("player-link")
    #list_thr = set([thr.get_attribute("href") for thr in list_plinks])
    
    main_table = browser.find_elements_by_id("player-table-statistics-body")[0]
    all_players = main_table.find_elements_by_class_name("pn")
    all_players_names = set([player.text.split('\n')[0] for player in all_players])
    all_players_dict = {name:[{},{},{},{}] for name in all_players_names}
    
    team_name = browser.find_element_by_tag_name("h1").text
    print(team_name)
    opt_tabels = browser.find_elements_by_id("team-squad-stats-options")[0]
    linked_tabels = opt_tabels.find_elements_by_xpath(".//a")[:-1] 
    
    iter=0
    for link_table in linked_tabels:
        link_table.click()
        link_table.parent.implicitly_wait(10)
        rel_table = browser.find_elements_by_id("team-squad-stats-"+link_table.text.lower())[0]
        table = rel_table.find_elements_by_id("top-player-stats-summary-grid")[0]
        team_header = [h.text for h in table.find_element_by_tag_name("thead").find_elements_by_tag_name("th")[7:]]
        team_body = table.find_element_by_tag_name("tbody")
        team_lines = team_body.find_elements_by_tag_name("tr")
        for line in team_lines:
            player_data_line = [l.text for l in line.find_elements_by_tag_name("td")[2:]]
            player_name = player_data_line[0].split('\n')[0]
            player_data_line = player_data_line[5:]
            all_players_dict[player_name][iter] = {h:d for h,d in izip(team_header,player_data_line) if h!="Discipline"}
        iter+=1
    
    all_teams_dict[team_name][3]=all_players_dict
    
            
if __name__ == '__main__':
    start_crawl() 