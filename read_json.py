#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import nltk
from unicodedata import normalize
import re
import matplotlib.pyplot as plt
import numpy as np
import codecs

#transforma as wtopwords de unicode para str
def getStopWord():
	stopwords = nltk.corpus.stopwords.words('portuguese')
	strStopWords = []
	for word in stopwords:
		strStopWords.append(convertUnicodeToString(word))

	return strStopWords

def remover_acentos(txt, codif='utf-8'):
	return normalize('NFKD', txt.decode(codif)).encode('ASCII','ignore')


def getVocabulario():
	with open('datasetNews.json') as json_data:
		data = json.load(json_data)
		tokenDataSet = set()
		for news in data:
			#Remove espaços duplicados e transforma string em lista de palavras
			token_news = set(re.sub(' +',' ',news['texto'].strip()).split())
			tokenDataSet = tokenDataSet.union(token_news)

	return tokenDataSet

def frequencyTokensInDataset():
	with open('datasetNews.json') as json_data:
		data = json.load(json_data)
		mapNews = {}
		for news in data:
			mapNews[news['id']] = getClearNews(re.sub(' +',' ',news['texto'].strip()).split())
		
	mapFrequencyTokens = {}
	for news in mapNews.values():
		for tokenNews in news:
			if(tokenNews != ""):
				if tokenNews in mapFrequencyTokens:
					mapFrequencyTokens[tokenNews]+=1
				else:
					mapFrequencyTokens[tokenNews]=1

	#Ordena mapa pelo valor
	mapPlot = {}
	k=100
	fileSaida = open('frequencyWords.txt','w')
	for element in sorted(mapFrequencyTokens.items(), key=lambda x: x[1])[-k:]:
		(word, frequency) = element
		mapPlot[word] = frequency
		lineFile = word+ ","+str(frequency)+"\n"
		fileSaida.write(lineFile)

	fig, ax = plt.subplots()
	plt.bar(range(k), mapPlot.values(), align='center')
	plt.xticks(range(k), mapPlot.keys())
	plt.title("Frenquencia dos tokens")
	plt.ylabel("frequency")

	#Rotacionar o label x
	for tick in ax.get_xticklabels():
		tick.set_rotation(90)

	plt.show()

	
def sizeDocumentDistribution():
	with open('datasetNews.json') as json_data:
		data = json.load(json_data)
		mapNewsTokens = {}
		for news in data:
			mapNewsTokens[news['id']] = len(getClearNews(re.sub(' +',' ',news['texto'].strip()).split()))

	print mapNewsTokens
	# size = len(mapNewsTokens)
	# plt.bar(size, mapNewsTokens.values(), align='center')
	# plt.xticks(size, mapNewsTokens.keys())
	# plt.show()

	plt.hist(mapNewsTokens.values())
	plt.title("Gaussian Histogram")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.show()


def convertUnicodeToString(token):
	token = normalize('NFKD', token).encode('ascii','ignore')
	return token

def getClearNews(news):
	stopWords = getStopWord()
	formattedNews = []

	for token in news:
		token = token.lower()
		token = convertUnicodeToString(token)
		token = removeEspecialChar(token)
		if(token not in stopWords):
			token = remover_acentos(token)	
			formattedNews.append(token)
	return formattedNews

def clearVocabulario(vocabulario):
	stopWords = getStopWord()
	clean_vocabulario = set()
	for token in vocabulario:
		token = token.lower()
		token = convertUnicodeToString(token)
		#Falar no relatório
		token = removeEspecialChar(token)
		if(token not in stopWords):
			token = remover_acentos(token)
			clean_vocabulario.add(token)
	return clean_vocabulario


def removeEspecialChar(token):
	char_esp = "?()!:;.-,'\""
	formatted_token=""
	for char in token:
		if char not in char_esp:
			formatted_token = formatted_token+char
	return formatted_token

if __name__ == "__main__":
	#Questao 2
	vocabulario = getVocabulario()
	print "Tamanho vocabulário ",len(vocabulario)

	#Questão 3
	cleanVocabulario = clearVocabulario(vocabulario)
	print "Tamanho vocabulário limpo ",len(cleanVocabulario)

	#Questão 4
	frequencyTokensInDataset()
	#sizeDocumentDistribution()
