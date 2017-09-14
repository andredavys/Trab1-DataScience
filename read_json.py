#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import nltk
from unicodedata import normalize
import re
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import math

#transforma as wtopwords de unicode para str
def getStopWord():
	stopwords = nltk.corpus.stopwords.words('portuguese')
	strStopWords = []
	for word in stopwords:
		strStopWords.append(convertUnicodeToString(word))

	return strStopWords

def removerAcentos(txt, codif='utf-8'):
	return normalize('NFKD', txt.decode(codif)).encode('ASCII','ignore')


def getVocabulario():
	with open('datasetNews.json') as json_data:
		data = json.load(json_data)
		tokenDataSet = set()
		for news in data:
			#Remove espaços duplicados e transforma string em lista de palavras
			token_news = set(re.sub(' +',' ',news['texto']).split())
			tokenDataSet = tokenDataSet.union(token_news)

	return tokenDataSet

def frequencyTokensInDataset():
	with open('datasetNews.json') as json_data:
		data = json.load(json_data)
		mapNews = {}
		for news in data:
			mapNews[news['id']] = getClearNews(re.sub(' +',' ',news['texto']).split())
		
	mapFrequencyTokens = {}
	for news in mapNews.values():
		for tokenNews in news:
			if tokenNews in mapFrequencyTokens:
				mapFrequencyTokens[tokenNews]+=1
			else:
				mapFrequencyTokens[tokenNews]=1

	#Ordena mapa pelo valor
	mapPlot = {}
	k=100
	for element in sorted(mapFrequencyTokens.items(), key=lambda x: x[1])[-k:]:
		(word, frequency) = element
		mapPlot[word] = frequency

	#print mapPlot
	#print sortMapFrequencyTokens[(size-10)][0], "->", sortMapFrequencyTokens[(size-10)][1]
	plt.bar(range(k), mapPlot.values(), align='center')
	plt.xticks(range(k), mapPlot.keys())
	plt.show()
	
def sizeDocumentDistribution():
	with open('datasetNews.json') as json_data:
		data = json.load(json_data)
		mapNewsTokens = {}
		for news in data:
			mapNewsTokens[news['id']] = len(getClearNews(re.sub(' +',' ',news['texto']).split()))

	#print mapNewsTokens
	# size = len(mapNewsTokens)
	# plt.bar(size, mapNewsTokens.values(), align='center')
	# plt.xticks(size, mapNewsTokens.keys())
	# plt.show()
	plt.hist(mapNewsTokens.values())
	plt.title("Gaussian Histogram")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.show()

def makeBagOfWords(vocabulario):

	i=0
	tokenID = {}
	for token in vocabulario:
		tokenID[token] = i
		i+=1

	with open('datasetNews.json') as json_data:
		data = json.load(json_data)
		newsInBagOfWords = {}

		for news in data:
			#gerar lista com 0's do tamanho do vocabulário
			vector = [0] * len(vocabulario)
			for tokenNews in getClearNews(re.sub(' +',' ',news['texto']).split()):
				vector[ tokenID[tokenNews] ] = 1

			newsInBagOfWords[news['id']] = vector

	return newsInBagOfWords,tokenID

def euclideanDistance2(p,q):
	size = len(p)
	distance = 0
	for i in range(size):
		distance += (p[i] - q[i])**2

	return distance

def distanceBetweenDocs(newsInBagOfWords,nameFile):
	file = open(nameFile,'w')
	sizeNews = len(newsInBagOfWords)
	for i in range(1,sizeNews+1):
		for j in range(i,sizeNews+1):
			p = newsInBagOfWords[i]
			q = newsInBagOfWords[j]
			distance = euclideanDistance2(p,q)
			print i,",",j," - ",distance
			file.write(str(i)+","+str(j)+" = "+str(distance)+"\n")
	file.close()


def buildAchiloptasMatrix(n,d):
	mult = math.sqrt(3/(n*1.0))
	#Cria matriz nxd com 0's
	achiloptasMatrix = []
	for i in range(n):
		achiloptasMatrix.append([0]*d)

	for i in range(n):
		for j in range(d):
			x = random.randint(1, 6)
			#prob 1/6
			if(x==1):
				achiloptasMatrix[i][j] = mult
			#prob 2/3
			elif(x>=2 and x<6):
				achiloptasMatrix[i][j] = 0
			#prob 1/6
			elif(x==6):
				achiloptasMatrix[i][j] = -mult
	return achiloptasMatrix

def buildGaussianMatrix(n,d):
	gaussianMatrix = []
	for i in range(n):
		gaussianMatrix.append([0]*d)

	for i in range(n):
		for j in range(d):
			gaussianMatrix[i][j] = random.uniform( 0,1/(n*1.0) )


	return gaussianMatrix




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
			token = removerAcentos(token)	
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
			token = removerAcentos(token)
			clean_vocabulario.add(token)
	return clean_vocabulario

def procedureQuestion7(vocabulary):
	d = len(vocabulary)

	for n in [4, 16, 64, 256, 1024, 4096]:

		#Step 1
		begin = time.time()
		achiloptasMatrix = buildAchiloptasMatrix(n,d)
		timeBuildAchiloptasMatrix = time.time()-begin

		#Step 2
		begin = time.time()
		gaussianMatrix = buildGaussianMatrix(n,d)
		timeBuildGaussianMatrix = time.time()-begin

def removeEspecialChar(token):
	char_esp = "?()!:;.,'\""
	formatted_token=""
	for char in token:
		if char not in char_esp:
			formatted_token = formatted_token+char
	return formatted_token


if __name__ == "__main__":
	#Questao 2
	# vocabulario = getVocabulario()
	# print "Tamanho vocabulário ",len(vocabulario)

	# #Questão 3
	# cleanVocabulario = clearVocabulario(vocabulario)
	# print "Tamanho vocabulário limpo ",len(cleanVocabulario)

	# #Questão 4
	# #frequencyTokensInDataset()
	# #sizeDocumentDistribution()

	# #Questão 5
	# newsInBagOfWords,tokenID = makeBagOfWords(cleanVocabulario)

	# #Questão 6
	# # inicio = time.time()
	# # distanceBetweenDocs(newsInBagOfWords,"distance_out.txt")
	# # fim = time.time()
	# # print "Tempo de execução para calcular distancias\n ",fim-inicio

	# #Questão 7
	print buildGaussianMatrix(10,2)