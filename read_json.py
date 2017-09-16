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
			if i!=j 
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

def transposeMatrix(matrixA):
	rows = len(matrixA)
	cols = len(matrixA[0])
	matrixT = [[0 for x in range(rows)] for y in range(cols)]
	for i in range(cols):
		for j in range(rows):
			matrixT[i][j] = matrixA[j][i]
	return matrixT

def multMatrix(X,Y):

	if(len(X[0])==len(Y)):
		result = [ [0]*len(Y[0]) for y in range(len(X)) ]
		for i in range(len(X)):
			for j in range(len(Y[0])):
				for k in range(len(Y)):
					result[i][j] += X[i][k] * Y[k][j]

		return result
	else:
		print "Multiplicação impossível"

#VERIFICAR SE É CORRETO FAZER ISTO
#calcula a diferença das distâncias euclidianas entre cada par de documentos
def calculateDistortion(fileData, fileProjected, fileOut):
	fileData = open(fileData, 'r')
	fileProjected = open(fileProjected, 'r')
	listRealData = fileData.readlines()
	listProjected = fileProjected.readlines()
	fileDistortion = open(fileOut, 'w')
	for i in range(len(listRealData)):
		realDistance = float(listRealData[i].split()[2])
		projectedDistance = float(listProjected[i].split()[2])
		distortion = 0
		if realDistance != 0:
			distortion = math.fabs(projectedDistance-realDistance) / realDistance
		fileDistortion.write(listRealData[i].split(",")[0]+ ","+listRealData[i].split(",")[1]+" = "+ str(distortion))

	fileData.close()
	fileProjected.close()
	fileDistortion.close()

def distortionByLemmaJL(dim, prob, size):
	e = math.sqrt(6*math.log10(2*size/prob)/dim)
	return e

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

def procedureQuestion7(vocabulary,newsInBagOfWords):
	d = len(vocabulary)

	for n in [4096]:# 64, 256, 1024, 4096]:

		#Step 1
		begin = time.time()
		achiloptasMatrix = buildAchiloptasMatrix(d,n)
		timeBuildAchiloptasMatrix = time.time()-begin

		#Step 2
		begin = time.time()
		gaussianMatrix = buildGaussianMatrix(d,n)
		timeBuildGaussianMatrix = time.time()-begin


		#Step 3
		newsRnEspaceAchiloptas = {}
		newsRnEspaceGaussian = {}
		begin = time.time()
		for idNews in newsInBagOfWords:
			print "Mut "
			# newsRnEspaceAchiloptas[idNews] = multMatrix([newsInBagOfWords[idNews]] ,achiloptasMatrix)[0]
			# newsRnEspaceGaussian[idNews] = multMatrix( [ newsInBagOfWords[idNews] ], gaussianMatrix)[0]
			newsRnEspaceAchiloptas[idNews] = np.matrix(newsInBagOfWords[idNews]).dot(np.matrix(achiloptasMatrix) )
			newsRnEspaceGaussian[idNews] = np.matrix(newsInBagOfWords[idNews]).dot(np.matrix(gaussianMatrix) )
			print "espaço\n"
		timeGenerateEspaceRn = time.time()-begin
		
		begin = time.time()
		distanceBetweenDocs(newsRnEspaceAchiloptas,"distance-Achiloptas-r"+str(n)+".txt")
		timeDistanceBetweenDocsAchiloptas = time.time()-begin

		begin = time.time()
		distanceBetweenDocs(newsRnEspaceGaussian,"distance-Gaussian-r"+str(n)+".txt")
		timeDistanceBetweenDocsGaussian = time.time()-begin

def removeEspecialChar(token):
	char_esp = "!#%&()*+-/[]\^_{}?:;`><123567890'.,&='\""
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

	# # #Questão 4
	# # #frequencyTokensInDataset()
	# # #sizeDocumentDistribution()

	# #Questão 5
	# newsInBagOfWords,tokenID = makeBagOfWords(cleanVocabulario)

	# #Questão 6
	# # inicio = time.time()
	# # distanceBetweenDocs(newsInBagOfWords,"distance_out.txt")
	# # fim = time.time()
	# # print "Tempo de execução para calcular distancias\n ",fim-inicio

	# #Questão 7
	# print buildGaussianMatrix(10,2)
	# procedureQuestion7(cleanVocabulario, newsInBagOfWords)
	# a = [[0 for x in range(3)] for y in range(2)]
	# b = [[0 for x in range(2)] for y in range(3)]

	# for i in range(2):
	# 	for j in range(3):
	# 		a[i][j] = random.randint(1,4)
	# 		b[j][i] = random.randint(1,4)
	print distortionByLemmaJL(2000,0.001,3000)

	# print a
	# print b
	# print multMatrix(a,b)
