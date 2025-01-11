import pandas, numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

dataset = pandas.read_csv('../../data/processed_tweets_140.csv', encoding='latin-1')
# Each feature_set is a tuple: (tweet, normalized_tweet, label)
feature_sets = list(zip(list(dataset.text.astype(str)), list(dataset.normalized_text.astype(str)), list(dataset.target)))
numpy.random.shuffle(feature_sets)
tweets, normalized_tweets, labels = list(zip(*feature_sets))
labels = [1 if label==0 else 0 for label in labels]

def extract_features():
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=7,
        max_df=0.8,
        stop_words=stopwords.words('english')
    )
    return vectorizer.fit_transform(list(dataset.normalized_text.astype(str))).toarray(), list(dataset.target)
