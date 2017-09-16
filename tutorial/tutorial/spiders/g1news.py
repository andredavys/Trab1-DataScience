#!/usr/bin/python
# -*- coding: UTF-8 -*-
import io
import json
import codecs
import time
from selenium import webdriver
from scrapy import Selector
import scrapy
from scrapy.item import Item, Field
from scrapy.http import TextResponse


class G1newsSpider(scrapy.Spider):
	name = 'g1news'
	allowed_domains = ['http://g1.globo.com/']
	start_urls = ['http://g1.globo.com/e-ou-nao-e/', 'http://g1.globo.com/economia/concursos-e-emprego/','http://g1.globo.com/natureza/']

	def __init__(self):
		ff_profile = webdriver.FirefoxProfile()
		self.driver = webdriver.Firefox(firefox_profile=ff_profile)	
		self.news = []

	def parse_news(self, response):
		noticia = {}
		print "PASSEI POR AQUI"
		noticia['titulo'] = response.css('.content-head__title::text').extract_first()

		yield noticia

	def parse(self, response):
		links = []
		for url in self.start_urls:
			time.sleep(3)
			self.driver.get(url)		

			i=0
			while i<60																																																																																																																																																																																																																																												:
				next = self.driver.find_element_by_class_name('load-more')
				try:
					next.click()
				
				except:
					print "\n\n#######\n","SAIU","\n\n#########\n"				
					break
				i=i+1    

			resp = TextResponse(url=self.driver.current_url, body=self.driver.page_source, encoding='utf-8')
			links = links + resp.css('.feed-post-link::attr(href)').extract()
			tam = len(links)
			print "\n\n************** ", tam, " **************\n\n"
			if tam>=1000:
				break
		

		c=1
		setLinks = set(links)
		n = len(setLinks)
		for link in setLinks:
			c+=1
			print "**********************",c/(n*1.0),"**********************","\n"
			
			self.driver.get(link)
			resp = TextResponse(url=self.driver.current_url, body=self.driver.page_source, encoding='utf-8')
			
			noticia = {
				
				'titulo': resp.css('.content-head__title::text').extract_first(),
				'subtitulo':resp.css('.content-head__subtitle::text').extract_first(),
				'texto':"".join(resp.css('.content-text__container::text').extract()),
				
			}

			if(noticia['titulo'] is not None and noticia['subtitulo'] is not None):
				self.news.append(noticia)

			yield noticia

		self.driver.close()

	def closed(self, reason):
		with io.open('noticiasG1.json', 'w', encoding='utf-8') as f:
			f.write(json.dumps(self.news, ensure_ascii=False))




	