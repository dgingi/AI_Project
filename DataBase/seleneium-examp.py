from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
import time

browser = webdriver.Chrome() # Get local session of firefox
browser.get("http://www.whoscored.com/Players/33404") # Load page
assert "Eden Hazard" in browser.title
elem = browser.find_element_by_id("player-tournament-stats-summary") # Find the query box
list_td = elem.find_elements_by_xptah(".//td")
browser.close()