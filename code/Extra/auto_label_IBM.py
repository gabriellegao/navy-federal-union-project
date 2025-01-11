# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:24:15 2020

@author: yugua
"""

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions, CategoriesOptions

natural_language_understanding = NaturalLanguageUnderstandingV1(                                         
    version='2018-11-16',
    iam_apikey= 'I4i4RrRXTtyqfFT5nteg0hIoEm2CgDN8qHc0AfeyPmfx',
    url='https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/ad9ebdfd-68db-45b6-8a76-494e22e7a8be'
    )
def Sentiment_score(input_text): 
    # Input text can be sentence, paragraph or document
    response = natural_language_understanding.analyze (
    text = input_text,
    features = Features(sentiment=SentimentOptions())).get_result()
    # From the response extract score which is between -1 to 1
    res = response.get('sentiment').get('document').get('score')
    return res

import pandas, os
dataset = pandas.read_csv(os.getcwd() + '/textblob_dataset.csv', encoding='utf8')
not_Navy_Tweet =  dataset['SenderUserId'] != 160344204010636
dataset = dataset[not_Navy_Tweet]
print(len(dataset))

dataset.sample(frac = 30000/len(dataset))
score = []
print(len(dataset))
for index, row in dataset.iterrows():
    print(index)
    try:
        i_score = Sentiment_score(row['NormalizedMessage'])
        score.append(i_score)
    except:
        score.append(0)

        
dataset['Label'] = score

print(dataset.describe())

neg = 0
neu = 0
pos = 0
for index, row in dataset.iterrows():
    if row['Label'] > 0:
        pos += 1
    elif row['Label'] == 0:
        neu += 1
    else:
        neg += 1
print("neg: ", neg, " pos: ", pos, " neu: ", neu)
dataset.to_csv(os.getcwd() + '/IBM_30k_sample.csv', index=False)