#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import nltk
from unicodedata import normalize

def remover_acentos(txt, codif='utf-8'):
	return normalize('NFKD', txt.decode(codif)).encode('ASCII','ignore')


def getVocabulario():
	with open('noticiasG1.json') as json_data:
		data = json.load(json_data)
		tokenDataSet = set()
		for news in data:
			token_news = set(news['titulo'].split()+news['subtitulo'].split()+news['texto'].split())
			tokenDataSet = tokenDataSet.union(token_news)

	return tokenDataSet

def convertUnicodeToString(token):
	token = normalize('NFKD', token).encode('ascii','ignore')
	return token

def clearVocabulario(vocabulario):
	stopWords = nltk.corpus.stopwords.words('portuguese')
	clean_vocabulario = set()
	for token in vocabulario:
		if(token not in stopWords):
			token = token.lower()
			token = convertUnicodeToString(token)
			token = removeEspecialChar(token)
			# print "novo token: ", len(token), token
			token = remover_acentos(token)
			clean_vocabulario.add(token)
	return clean_vocabulario


def removeEspecialChar(token):
	char_esp = "!#&()*+-/[]\^_{}?:;`><'.,&='\""
	formatted_token=""
	for char in token:
		if char not in char_esp:
			formatted_token = formatted_token+char
	return formatted_token

if __name__ == "__main__":
	#Questao 2
	vocabulario = getVocabulario()
	print "Tamanho vocabulário ",len(vocabulario)

	# #Questão 
	clean_vocabulario = clearVocabulario(vocabulario)
	print "Tamanho vocabulário limpo ",len(clean_vocabulario)	
