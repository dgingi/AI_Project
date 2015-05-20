from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

browser = webdriver.Chrome() # Get local session of Chrome
browser.get("http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/3853/Stages/7794/Fixtures/England-Premier-League-2013-2014") # Load page
browser.implicitly_wait(15)

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
    finally:
        pass
print games_by_month

home_table = browser.find_element_by_id("live-player-home-stats")
home_players = home_table.find_elements_by_class_name("pn")
home_players_names = set([player.text.split('\n')[0] for player in home_players])
home_players_dict = {name:[{},{},{},{}] for name in home_players_names}
away_table = browser.find_element_by_id("live-player-away-stats")
away_players = away_table.find_elements_by_class_name("pn")
away_players_names = set([player.text.split('\n')[0] for player in away_players])
away_players_dict = {name:[{},{},{},{}] for name in away_players_names}