import json, pandas, re
from nltk.tokenize import TweetTokenizer
import numpy
from sklearn.feature_extraction.text import CountVectorizer

tweet_tok = TweetTokenizer()


######################################################
###########  TEXT NORMALIZATION METHODS    ###########
######################################################

# Get contraction map
with open('../resources/contraction_map.json') as f:
    contraction_map = json.load(f)
    # Build regexps for cleaning data
    contraction_df = pandas.DataFrame.from_dict(contraction_map, orient='index', columns=['expanded'])
    contraction_df = contraction_df.assign(n=contraction_df.index.map(lambda e: e.count("'")))
    contraction_df = contraction_df.sort_values(by=['n'], ascending=False)
    contraction_re = re.compile('|'.join(['(?:\'|’)'.join(k.split("'")) for k in contraction_df.index]), re.IGNORECASE)

# Takes a string and returns a version with its contractions expanded
def expand_contractions(text):
    return re.sub(contraction_re, lambda c: c[0][0] + contraction_map.get(c[0].replace("’", "'"))[1:], text)

url_re = re.compile(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+|www.[^ ]+',re.VERBOSE | re.IGNORECASE)
# Strips URLs from the text provided as input
def clean_urls(text):
    return re.sub(url_re, '', text)

# Intended to normalize the string text using the above methods
# Converts the text to lower case, strips URLs, expands contractions, removes
# any hashtags, mentions, and special symbols
# TODO - number normalizer? probably quite frequent with credit union posts
# when it comes to percentages/rates/costs etc
def normalize_text(text):
    # normalize by converting to lowercase
    text = text.lower()
    text = clean_urls(text)
    #text = clean_numbers(text)
    text = expand_contractions(text)
    text = re.sub(r'@[A-Za-z0-9]+', " ", text)
    text = re.sub(r'#[A-Za-z0-9]+', " ", text)
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    return ' '.join(tweet_tok.tokenize(letters_only))

def normalize_docs(docs):
    return [normalize_text(doc) for doc in docs]

######################################################
###########     DATA COMPILER METHODS    #############
######################################################

def load_twitter_data():
    return pandas.read_excel('../data/Twitter Public Posts for Cornell Project 2016-2020.xlsx', encoding='utf8')

def load_facebook_data():
    return pandas.read_excel('../data/Facebook Public Posts for Cornell Project 2016-2020.xlsx', encoding='utf8')

def load_original_dataset():
    return pandas.concat([load_twitter_data(),load_facebook_data()])

def compile_normalized_dataset():
    df = load_original_dataset()
    df = df.assign(NormalizedMessage = normalize_docs(df.Message.astype(str)))
    df.to_csv('../data/dataset.csv', index=False)
    return df

def load_normalized_dataset():
    print("Attempting to load normalized dataset...")
    try:
        df = pandas.read_csv('../data/dataset.csv')
        print("Returning normalized dataset.")
    except FileNotFoundError as e:
        print("File not found - dataset hasn't yet been normalized.")
        print("Normalizing dataset...")
        df = data.compile_normalized_dataset()
        print("Dataset has been normalized")
    return df

def extract_features(docs, stop_words=None, ngram_range=(1,1)):
    count_vec = CountVectorizer(binary=True,
                                max_features=1000,
                                stop_words=stop_words,
                                ngram_range=ngram_range)
    doc_term_mat = count_vec.fit_transform(docs)
    co_occ_mat = numpy.dot(doc_term_mat.T, doc_term_mat)
    return doc_term_mat, co_occ_mat, count_vec
