#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json


def getVocabulario():
	with open('noticiasG1.json') as json_data:
	data = json.load(json_data)
	tokenDataSet = set()
	for news in data:
		token_news = set(news['titulo'].split()+news['subtitulo'].split()+news['texto'].split())
		tokenDataSet = tokenDataSet.union(token_news)

	print "Tamanho vocabulário ",len(tokenDataSet)
	return tokenDataSet



if __name__ == "__main__":
    main()