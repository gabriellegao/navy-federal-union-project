import pandas
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import normalize_text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Load airline dataset
# Normalize tweets
# Return two lists, one with normalized_tweets and the other with their labels
def load_airline_data():
    data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
    df = pandas.read_csv(data_source_url)
    features = df.iloc[:, 10].values
    df=df.assign(complaint=df.airline_sentiment.apply(lambda r: 1 if r=="negative" else 0))
    labels = df.complaint
    normalized_features = [normalize_text(tweet) for tweet in features]
    return normalized_features, labels

def train_classifier(features, labels):
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=7,
        max_df=0.8,
        stop_words=stopwords.words('english')
    )
    processed_features = vectorizer.fit_transform(features).toarray()
    classifier = MultinomialNB()
    classifier.fit(processed_features, labels)
    return classifier, vectorizer

def label_nfcu_tweets(classifier, vectorizer):
    nfcu = pandas.read_csv('../../data/dataset.csv')
    predictions = classifier.predict(vectorizer.transform(nfcu.NormalizedMessage.astype(str)))
    return nfcu.assign(IsComplaint=predictions)
