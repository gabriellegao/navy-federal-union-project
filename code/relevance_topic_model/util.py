import re, pickle, numpy, pandas, nltk, time, os, tqdm
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split

from gensim.models.phrases import Phrases, Phraser
from gensim.corpora.dictionary import Dictionary
from gensim import models

from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional,LSTM,Dense,Embedding,Dropout,Activation,Softmax
from keras.utils import np_utils

from tqdm import tqdm

EPOCHS = 3
TEST_SIZE = 0.5
N_LABELS = 2
CORPUS_PATH = '../../data/corpus.pkl'
# pickled python list of strings
RELEVANCE_LABELED_DATA_PATH = '../../data/manually_labeled_relevance_feature_sets.pkl'
# pickled python list of tuples (string, label)


W2V_PATH = 'w2v/w2v.model'
# try to load W2V from this constant path.
# if we can't, train the w2v model and save it there.
RELEVANCE_CLASSIFIER_PATH = 'relevance_classifier.h5'
# try to load classifier from this constant path.
# if we can't, train the relevance classifier and save it there.


##########################################################

def build_relevance_classifier(path='relevance_classifier.h5', test_size=0.5):
    with open(RELEVANCE_LABELED_DATA_PATH, mode="rb") as f:
        docs, labels = list(zip(*pickle.load(f)))
        corpus = Corpus.load()
        corpus.fit_w2v('w2v_models/w2v.model')
        return corpus.build_classifier(docs, labels, path=path, test_size=test_size)


############################################################

stopwords = list(nltk.corpus.stopwords.words('english'))
url_re = re.compile(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+|www.[^ ]+',re.VERBOSE | re.IGNORECASE)
non_alphanumeric_token_re = re.compile(r'\s[^a-zA-Z0-9\.,!?\'\$%]+\s|^[^a-zA-Z0-9\.,!?\'\$%]+$|^[^a-zA-Z0-9\.,!?\'\$%]+\s|\s[^a-zA-Z0-9\.,!?\'\$%]+$')

def dummy_fun(doc):
    return doc

##########################################################
##########################################################
##########################################################
class Corpus:
    def tokenize(self, raw_post):
        if type(raw_post)==list:
            return [self.tokenize(raw_post[i]) for i in tqdm(range(len(raw_post)))]
        # Remove URLs and mentions
        post = re.sub(url_re, ' URL ', raw_post)
        post = post.replace('â\x80\x99', "'").replace('’', "'")
        tokens = [e for e in re.sub(non_alphanumeric_token_re, ' NON_ALPHA_NUMERIC ', post).lower().split(' ') if e!=' ']
        return tokens

    # loads corpus from a filepath to a pickled python list of strings
    def load(path='../../data/corpus.pkl'):
        with open(path, mode="rb") as f:
            docs = pickle.load(f)
        corp = Corpus(docs, tokenize=True)
        return corp

    def __init__(self, docs, tokenize=False):
        if tokenize==True:
            docs = [self.tokenize(d) for d in docs]
        self.docs = docs

    def extract_features(self, docs):
        return  numpy.array(self.w2v.transform(docs))

    def fit_w2v(self,path=None, embedding_dim=2000, size=200):
        w2v = Word2Vec(tokenize=None, embedding_dim=embedding_dim, size=size)
        if not(w2v.load(path)):
            docs = self.docs
            w2v.fit(docs)
        self.w2v = w2v
        if path!=None:
            w2v.save(path)
        self.w2v = w2v
        return w2v

    def pmi(self, tokenize=None):
        # Need relevance filter first
        pass

    def compute_token_counts(self, ngram_range=(1,1), min_df=10, stop_words=stopwords):
        cvec = CountVectorizer(analyzer='word',
            tokenizer=dummy_fun,
            preprocessor=dummy_fun,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words = stop_words)
        X = cvec.fit_transform(self.docs)
        X[X > 0] = 1
        doc_term_mat = X
        doc_term_mat_counts = cvec.fit_transform(self.docs)

        tokens = cvec.get_feature_names()
        freqs = list(numpy.asarray(X.sum(axis=0))[0])
        # Make dict mapping token to frequency of occurrence, i.e. how many sentences it occurs in
        token_freq = dict(zip(tokens, freqs))
        # Make dict mapping token to probability of occurrence
        total = len(self.docs)
        token_prob = dict(zip(tokens, [freq/total for freq in freqs]))
        return Counter(token_freq).most_common(), Counter(token_prob).most_common()
        #self.token_com_freq = dict(zip(tokens, ))

    def extract_bigrams(self, phrase_filter=lambda e: re.match(r'[a-z\s]+$', e) and len([t for t in e.split(' ') if t in stopwords])==0, keywords=[]):
        p = Phrases(self.docs, min_count=1, threshold=1)
        #bigram = Phraser(phrases)
        phrases = [(phrase.decode('utf8'), score) for phrase, score in p.export_phrases(self.docs) if phrase_filter(phrase.decode('utf8'))]
        #bigrams = bigram[self.w2v.vocab.keys()]
        phrases = list(set(phrases))
        phrases.sort(key=lambda e: e[1])
        if keywords!=None and len(keywords)!=0:
            return [phrase for phrase in phrases if len([e for e in phrase[0].split(' ') if e in keywords])!=0]
        return phrases

    def build_classifier(self, docs, labels, path='relevance_classifier.h5', test_size=0.5, tokenize=True):

        n_labels = len(set([e for e in labels]))
        model = Model(self, n_labels)
        # Extract features
        if tokenize==True:
            X = numpy.array(self.w2v.transform([self.tokenize(d) for d in docs]))
        else:
            X = numpy.array(self.w2v.transform(docs))
        y = numpy.array(np_utils.to_categorical(labels))
        # Fit model
        if not(os.path.exists(path)):
            model.fit(X, y, test_size=test_size)
        return model

    def compute_pmi(self, vocab):
        #TODO
        pass


class Word2Vec:
    # docs should be a list of tokenized strings (lists like ["he", "went", "to"])
    def __init__(self, tokenize=None, embedding_dim=2000, size=200):
        self.embedding_dim = embedding_dim
        self.tokenize=tokenize
        self.size = size

    def load(self, path):
        try:
            #print("Attempting to load " + path + "...")
            self.model = models.Word2Vec.load(path)
            self.update()
            return self.model
        except:
            #print("Could not load word2vec model from '" + path + "'.")
            return False

    def save(self, path):
        self.model.save(path)

    def fit(self, X):
        start = time.time()
        model = models.Word2Vec(X,
            size=2000,
            min_count=5,
            window=5)
        self.model = model
        self.update()

    def update(self):
        start = time.time()
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(self.model.wv.vocab.keys(), allow_update=True)
        vocab = {v: k + 1 for k, v in gensim_dict.items()}
        self.vocab = vocab

        embedding_weights = numpy.zeros((len(self.model.wv.vocab)+1, self.embedding_dim))
        for word, idx in tqdm(vocab.items()):
            embedding_vector = self.model[word]
            if embedding_vector is not None:
                embedding_weights[idx] = embedding_vector
        self.embedding_weights = embedding_weights
        print("word2vec took " + str(time.time()-start) + " seconds to complete update of vocab and embedding weights.")


    def fit_old(self, X, y=None):
        start = time.time()
        model = models.Word2Vec(X,
            size=2000,
            min_count=5,
            window=5)
        self.model = model

        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        vocab = {v: k + 1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in vocab.keys()}
        n_vocabs = len(vocab) + 1
        embedding_weights = numpy.zeros((n_vocabs, self.embedding_dim))
        for w, index in vocab.items():
            embedding_weights[index, :] = w2vec[w]
        self.vocab = vocab
        self.embedding_weights = embedding_weights
        print("word2vec took " + str(time.time()-start) + " seconds to complete.")

    def transform(self, X):
        vectorized_docs = []
        for doc in X:
            if self.tokenize!=None:
                doc = self.tokenize(doc)
            doc = [self.vocab.get(token,0) for token in doc]
            if len(doc)>self.size:
                doc = doc[-self.size:]
            elif len(doc)<self.size:
                doc = [0]*(self.size-len(doc))+doc
            vectorized_docs.append(doc)
        return vectorized_docs

class Model:
    def __init__(self,corpus,n_labels, path='relevance_classifier.h5'):
        self.corpus = corpus
        self.vocab = corpus.w2v.vocab
        #self.embedding_weights = corpus.w2v.embedding_weights
        self.Embedding_dim = corpus.w2v.embedding_dim
        self.n_labels = n_labels
        self.maxlen = corpus.w2v.size
        self.path = path
        #w2v_folder = self.w2v.path.split('/')[0]
        #self._trained = os.path.exists(w2v_folder + '/' + path)
        model = Sequential()
        #input dim(140,100)
        model.add(Embedding(output_dim = self.Embedding_dim,
                           input_dim=len(self.vocab)+1,
                           weights=[corpus.w2v.embedding_weights],
                           input_length=self.maxlen))
        model.add(Bidirectional(LSTM(50),merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_labels))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
        model.summary()
        self.model = model

    def extract_features(self, docs, labels):
        X = numpy.array(self.corpus.w2v.transform([self.corpus.tokenize(d) for d in docs]))
        y = numpy.array(np_utils.to_categorical(labels))
        return X, y

    def fit(self, docs, labels, epochs=5, test_size=0):
        X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=test_size)
        self.model.fit(X_train, y_train, batch_size=32, epochs=epochs,
                      validation_data=(X_test, y_test))
        #w2v_folder = self.w2v.path.split('/')[0]
        self.model.save(self.path)

    def predict(self,docs):
        #if type(docs[0])==str:
            #docs = self.tokenize(docs)
        model = self.model
        #w2v_folder = self.w2v.path.split('/')[0]
        model.load_weights(self.path)
        vectorized_docs = numpy.array(self.corpus.w2v.transform([self.corpus.tokenize(d) for d in docs]))
        return model.predict(vectorized_docs)
