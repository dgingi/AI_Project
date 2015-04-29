import scrapy
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.http import Request
from tut.items import TutItem as Item
 
from selenium import webdriver
 
class SeleniumSpider(CrawlSpider):
    name = "dmoz"
    start_urls = ["http://www.whoscored.com/Players/33404"]
 
    rules = (
        Rule(SgmlLinkExtractor(allow=('\.html', )), callback='parse_page',follow=True),
    )
 
    def __init__(self):
        CrawlSpider.__init__(self)
        self.verificationErrors = []
        self.selenium = webdriver.Chrome()
 
    def __del__(self):
        self.selenium.stop()
        print self.verificationErrors
        CrawlSpider.__del__(self)
 
    def parse_page(self, response):
        item = Item()
 
        #hxs = HtmlXPathSelector(response)
        #Do some XPath selection with Scrapy
        #hxs.select('//div').extract()
 
        sel = self.selenium
        sel.get(response.url)
 
        #Wait for javscript to load in Selenium
        time.sleep(2.5)
 
        #Do some crawling of javascript created content with Selenium
        print (sel.find_element_by_id("top-player-stats-summary-grid").text())
        yield item
 
