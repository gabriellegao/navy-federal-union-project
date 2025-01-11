# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:25:52 2020

@author: yugua
"""

import pandas, os
import numpy
from textblob import TextBlob

dataset = pandas.read_csv(os.path.join(os.getcwd(), "../..") + '/data/dataset.csv', encoding='utf8')
corpus = dataset.NormalizedMessage.values.astype('U')

from textblob import TextBlob

polarity = []
subjectivity = []
for document in corpus:
    blob = TextBlob(document)
    print(blob.sentiment)
    print(blob.sentiment[0])
    print(blob.sentiment[1])
    polarity.append(blob.sentiment[0])
    subjectivity.append(blob.sentiment[1])
    count += 1
        
dataset['Polarity'] = polarity
dataset['Subjectivity'] = subjectivity
print('Done')

dataset.to_csv(os.getcwd() + '/textblob_dataset.csv', index=False)
