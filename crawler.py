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
    browser.get("http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/3853/Stages/7794/Fixtures/England-Premier-League-2013-2014") # Load page
    browser.implicitly_wait(15)
    parse_league(browser)
    
    browser.close()
    
def parse_league(browser):
    """
    Start parsing the league table.
    
    
    TODO: parse the other tables
    """

    disp_month = browser.find_element_by_id('date-controller')
    prev_month = disp_month.find_elements_by_tag_name('a')[0]
    months = ['May','Apr','Mar','Feb','Jan','Dec','Nov','Oct','Sep','Aug']
    games_by_month = {i:None for i in months}
    wait = WebDriverWait(browser,15)

    for i in months:
        try:
            wait.until(EC.text_to_be_present_in_element((By.CLASS_NAME,"rowgroupheader"),i))
            games_by_month[i] = browser.find_elements_by_xpath('//a[@class="result-1 rc"]')
            prev_month.click()
        finally: #TODO - catch the TimeOut exception and do something about it
            pass
    
    for month in games_by_month:
        for game in month:
            parse_game(browser,game.get_attribute("href"))    
    
    #list_tlinks = browser.find_elements_by_class_name("team-link")
    #list_thr = set([thr.get_attribute("href") for thr in list_tlinks])
    
    #all_teams_names = set([thr.text for thr in list_tlinks])
    #all_teams_dict = {name:[{},{},{},{}] for name in all_teams_names}
    
    #opt_tabels = browser.find_elements_by_id("stage-team-stats-options")[0]
    #linked_tabels = opt_tabels.find_elements_by_xpath(".//a")[:-1] 
    
    #iter=0 
    
    #for link_table in linked_tabels:
    #    link_table.click()
    #    link_table.parent.implicitly_wait(10)
    #    rel_table = browser.find_elements_by_id("stage-team-stats-"+link_table.text.lower())[0]
    #    table = rel_table.find_elements_by_id("top-team-stats-summary-grid")[0]
    #    league_header = [h.text for h in table.find_element_by_tag_name("thead").find_elements_by_tag_name("th")[2:]]
    #    league_body = table.find_element_by_tag_name("tbody")
    #    league_lines = league_body.find_elements_by_tag_name("tr")
    #    for line in league_lines:
    #        team_data_line = [l.text for l in line.find_elements_by_tag_name("td")[1:]]
    #        team_name = team_data_line[0]
    #        team_data_line = team_data_line[1:]
    #        all_teams_dict[team_name][iter] = {h:d for h,d in izip(league_header,team_data_line) if h!="Discipline"}
    #    iter+=1
           
    #for link in list_thr:
    #    parse_team(browser,link,all_teams_dict)
    
    #with open("output.pckl",'w') as output:
    #    dump(all_teams_dict, output)

    
def parse_game(browser,link,all_teams_dict):
    """
    Start parsing a single game (Home VS Away).
    
    TODO: parse the other tables
    """
    
    browser.get(link)
    browser.implicitly_wait(10)
    players_stats_elm = browser.find_element_by_link_text("Player Statistics")
    players_stats_elm.click() #TODO - see if need to wait for page to load
    
    #list_plinks = browser.find_elements_by_class_name("player-link")
    #list_thr = set([thr.get_attribute("href") for thr in list_plinks])
    
    home_table = browser.find_element_by_id("live-player-home-stats")
    home_players = home_table.find_elements_by_class_name("pn")
    home_players_names = set([player.text.split('\n')[0] for player in home_players])
    home_players_dict = {name:[{},{},{},{}] for name in home_players_names}
    away_table = browser.find_element_by_id("live-player-away-stats")
    away_players = away_table.find_elements_by_class_name("pn")
    away_players_names = set([player.text.split('\n')[0] for player in away_players])
    away_players_dict = {name:[{},{},{},{}] for name in away_players_names}
    
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