import json, pandas, re, numpy, enchant
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
######################################################
###########  TEXT NORMALIZATION METHODS    ###########
######################################################

english_dict = enchant.Dict("en_US")

# Get contraction map
with open('resources/contraction_map.json') as f:
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

def valid_words(text):
    return ' '.join([english_dict.check(w) for w in text.split(' ')])


######################################################
###########  DATA NORMALIZATION METHODS    ###########
######################################################
def normalize_dataset(df):
    df = df.assign(NormalizedMessage = df.Message.astype(str).apply(normalize_text))#normalize_docs(df.Message.astype(str)))
    return df.assign(EnglishWords = df.NormalizedMessage.astype(str).apply(valid_words))

def load_data(data_path='../../data/'):
    print("Attempting to load normalized dataset...")
    try:
        df = pandas.read_csv(data_path+'dataset.csv')
        print("Returning normalized dataset.")
    except FileNotFoundError as e:
        print("File not found - dataset hasn't yet been normalized.")
        print("Normalizing dataset...")
        df = pandas.concat([
            pandas.read_excel(data_path+'Twitter Public Posts for Cornell Project 2016-2020.xlsx', encoding='utf8'),
            pandas.read_excel(data_path+'Facebook Public Posts for Cornell Project 2016-2020.xlsx', encoding='utf8')
        ])
        df = df.assign(NormalizedMessage = normalize_docs(df.Message.astype(str)))
        df.to_csv(data_path+'dataset.csv', index=False)
        print("Dataset has been normalized")
    return df

def shuffle(df):
    # returns the DataFrame with the rows shuffled randomly
    pass

######################################################
###########  FEATURE EXTRACTION METHODS    ###########
######################################################

def extract_freqs(docs, stop_words=None, ngram_range=(1,1), max_features=1000, min_df=0, max_df=1.0):
    count_vec = CountVectorizer(binary=True,
                                max_features=max_features,
                                stop_words=stop_words,
                                ngram_range=ngram_range,
                                min_df=min_df,
                                max_df=max_df)
    doc_term_mat = count_vec.fit_transform(docs)
    co_occ_mat = numpy.dot(doc_term_mat.T, doc_term_mat)
    return doc_term_mat, co_occ_mat, count_vec

def extract_tfidf(docs, stop_words=None, ngram_range=(1,1), max_features=1000, min_df=0, max_df=1.0):
    tfidf_vec = TfidfVectorizer(binary=True,
                                max_features=max_features,
                                stop_words=stop_words,
                                ngram_range=ngram_range,
                                min_df=min_df,
                                max_df=max_df)
    doc_term_mat = tfidf_vec.fit_transform(docs)
    co_occ_mat = numpy.dot(doc_term_mat.T, doc_term_mat)
    return doc_term_mat, co_occ_mat, tfidf_vec

######################################################
###########  EXPLORATORY ANALYSIS METHODS  ###########
######################################################

def compute_top_term_freqs():
    pass
