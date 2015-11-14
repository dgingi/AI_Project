'''
Created on Nov 14, 2015

@author: root
'''
import unittest
from selenium import webdriver
from crawler import _get_played_months
from calendar import month_abbr as _months # == ['','Jan','Feb',...,'Nov','Dec']
from time import sleep


class TestCrawlerFunctionality(unittest.TestCase):


    def setUp(self):
        chop = webdriver.ChromeOptions()
        chop.add_extension('../AdBlock_v2.36.2.crx')
        self.driver = webdriver.Chrome(chrome_options=chop)
        pass


    def tearDown(self):
        self.driver.quit()
        pass


    def test_get_played_months(self):
        self.fix_links = {'PL_2010':'http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/2458/Stages/4345/Fixtures/England-Premier-League-2010-2011',\
                          'PL_2011':'http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/2935/Stages/5476/Fixtures/England-Premier-League-2011-2012',\
                          'PL_2012':'http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/3389/Stages/6531/Fixtures/England-Premier-League-2012-2013',\
                          'PL_2013':'http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/3853/Stages/7794/Fixtures/England-Premier-League-2013-2014',\
                          'PL_2014':'http://www.whoscored.com/Regions/252/Tournaments/2/Seasons/4311/Stages/9155/Fixtures/England-Premier-League-2014-2015',\
                          'SA_2010':'http://www.whoscored.com/Regions/108/Tournaments/5/Seasons/2626/Stages/4659/Fixtures/Italy-Serie-A-2010-2011',\
                          'SA_2011':'http://www.whoscored.com/Regions/108/Tournaments/5/Seasons/3054/Stages/5667/Fixtures/Italy-Serie-A-2011-2012',\
                          'SA_2012':'http://www.whoscored.com/Regions/108/Tournaments/5/Seasons/3512/Stages/6739/Fixtures/Italy-Serie-A-2012-2013',\
                          'SA_2013':'http://www.whoscored.com/Regions/108/Tournaments/5/Seasons/3978/Stages/8019/Fixtures/Italy-Serie-A-2013-2014',\
                          'SA_2014':'http://www.whoscored.com/Regions/108/Tournaments/5/Seasons/5441/Stages/11369/Fixtures/Italy-Serie-A-2014-2015',\
                          'LL_2010':'http://www.whoscored.com/Regions/206/Tournaments/4/Seasons/2596/Stages/4624/Fixtures/Spain-La-Liga-2010-2011',\
                          'LL_2011':'http://www.whoscored.com/Regions/206/Tournaments/4/Seasons/3004/Stages/5577/Fixtures/Spain-La-Liga-2011-2012',\
                          'LL_2012':'http://www.whoscored.com/Regions/206/Tournaments/4/Seasons/3470/Stages/6652/Fixtures/Spain-La-Liga-2012-2013',\
                          'LL_2013':'http://www.whoscored.com/Regions/206/Tournaments/4/Seasons/3922/Stages/7920/Fixtures/Spain-La-Liga-2013-2014',\
                          'LL_2014':'http://www.whoscored.com/Regions/206/Tournaments/4/Seasons/5435/Stages/11363/Fixtures/Spain-La-Liga-2014-2015',\
                          'BL_2010':'http://www.whoscored.com/Regions/81/Tournaments/3/Seasons/2520/Stages/4448/Fixtures/Germany-Bundesliga-2010-2011',\
                          'BL_2011':'http://www.whoscored.com/Regions/81/Tournaments/3/Seasons/2949/Stages/5492/Fixtures/Germany-Bundesliga-2011-2012',\
                          'BL_2012':'http://www.whoscored.com/Regions/81/Tournaments/3/Seasons/3424/Stages/6576/Fixtures/Germany-Bundesliga-2012-2013',\
                          'BL_2013':'http://www.whoscored.com/Regions/81/Tournaments/3/Seasons/3863/Stages/7806/Fixtures/Germany-Bundesliga-2013-2014',\
                          'BL_2014':'http://www.whoscored.com/Regions/81/Tournaments/3/Seasons/4336/Stages/9192/Fixtures/Germany-Bundesliga-2014-2015',\
                          'L1_2010':'http://www.whoscored.com/Regions/74/Tournaments/22/Seasons/2417/Stages/4273/Fixtures/France-Ligue-1-2010-2011',\
                          'L1_2011':'http://www.whoscored.com/Regions/74/Tournaments/22/Seasons/2920/Stages/5451/Fixtures/France-Ligue-1-2011-2012',\
                          'L1_2012':'http://www.whoscored.com/Regions/74/Tournaments/22/Seasons/3356/Stages/6476/Fixtures/France-Ligue-1-2012-2013',\
                          'L1_2013':'http://www.whoscored.com/Regions/74/Tournaments/22/Seasons/3836/Stages/7771/Fixtures/France-Ligue-1-2013-2014',\
                          'L1_2014':'http://www.whoscored.com/Regions/74/Tournaments/22/Seasons/4279/Stages/9105/Fixtures/France-Ligue-1-2014-2015'}
        
        self.res =  {'PL_2010':_months[1:6][::-1]+_months[8:13][::-1],\
                          'PL_2011':_months[1:6][::-1]+_months[8:13][::-1],\
                          'PL_2012':_months[1:6][::-1]+_months[8:13][::-1],\
                          'PL_2013':_months[1:6][::-1]+_months[8:13][::-1],\
                          'PL_2014':_months[1:6][::-1]+_months[8:13][::-1],\
                          'SA_2010':_months[1:6][::-1]+_months[8:13][::-1],\
                          'SA_2011':_months[1:6][::-1]+_months[9:13][::-1],\
                          'SA_2012':_months[1:6][::-1]+_months[8:13][::-1],\
                          'SA_2013':_months[1:6][::-1]+_months[8:13][::-1],\
                          'SA_2014':_months[1:6][::-1]+_months[8:13][::-1],\
                          'LL_2010':_months[1:6][::-1]+_months[8:13][::-1],\
                          'LL_2011':_months[1:6][::-1]+_months[8:13][::-1],\
                          'LL_2012':_months[1:7][::-1]+_months[8:13][::-1],\
                          'LL_2013':_months[1:6][::-1]+_months[8:13][::-1],\
                          'LL_2014':_months[1:6][::-1]+_months[8:13][::-1],\
                          'BL_2010':_months[1:6][::-1]+_months[8:13][::-1],\
                          'BL_2011':_months[1:6][::-1]+_months[8:13][::-1],\
                          'BL_2012':_months[1:6][::-1]+_months[8:13][::-1],\
                          'BL_2013':_months[1:6][::-1]+_months[8:13][::-1],\
                          'BL_2014':_months[1:6][::-1]+_months[8:13][::-1],\
                          'L1_2010':_months[1:6][::-1]+_months[8:13][::-1],\
                          'L1_2011':_months[1:6][::-1]+_months[8:13][::-1],\
                          'L1_2012':_months[1:6][::-1]+_months[8:13][::-1],\
                          'L1_2013':_months[1:6][::-1]+_months[8:13][::-1],\
                          'L1_2014':_months[1:6][::-1]+_months[8:13][::-1]}
        
        for league in self.fix_links:
            self.driver.get(self.fix_links[league])
            played_years , played_months = _get_played_months(self.driver)
            self.assertEqual(len(played_years), 2)
            self.assertListEqual(played_months, self.res[league])
            sleep(5)
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()