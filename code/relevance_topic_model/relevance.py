import pickle, time, numpy
from util import Corpus
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.decomposition import TruncatedSVD
import seaborn as sns

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

stopwords = list(nltk.corpus.stopwords.words('english')) + ["non_alpha_numeric"]
ignore_tokens = ["non_alpha_numeric"]

def build_classifier(rel_classifier_path=RELEVANCE_CLASSIFIER_PATH, test_size=0.5):
    with open(RELEVANCE_LABELED_DATA_PATH, mode="rb") as f:
        docs, labels = list(zip(*pickle.load(f)))
        corpus = Corpus.load()
        corpus.fit_w2v(W2V_PATH)
        return corpus.build_classifier(docs, labels, path=rel_classifier_path, test_size=test_size)

def label_data(classifier, chunk_size=1000, n=100000, corpus_path=CORPUS_PATH):
    start = time.time()
    with open(corpus_path, mode="rb") as f:
        docs = pickle.load(f)

    y = []
    n = min(n, len(docs))
    start_idxs = list(range(0, n, chunk_size))
    for i in tqdm(range(len(start_idxs))):
        start = start_idxs[i]
        if start+chunk_size>n:
            end = n
        else:
            if start+chunk_size==n:
                end = n
            else:
                end = start_idxs[i+1]
        chunk = docs[start:end]
        pred = classifier.predict(chunk).tolist()
        y+=pred
    return list(zip(docs, y))


def lemmatize(tokens):
    lemmatized_tokens = []
    for token, tag in nltk.pos_tag([t for t in tokens if t]):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        #stemmer = PorterStemmer()
        #token = stemmer.stem(token)
        if len(token)>0:#and token.lower() not in stopwords:
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos))
    return lemmatized_tokens

def dummy_fun(doc):
    return doc

class TopicMiner:
    def __init__(self, n_topics=10, n_words=10):
        self.n_topics = 10
        self.n_words = 10

    def preprocess(self, docs):
        lemmatized_docs = [lemmatize(doc) for doc in docs]
        vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=dummy_fun,
            preprocessor=dummy_fun,
            token_pattern=None,
            ngram_range=(1,2),
            stop_words = ignore_tokens
        )
        doc_term_mat = vectorizer.fit_transform(lemmatized_docs)
        vocab = vectorizer.get_feature_names()
        return lemmatized_docs, vectorizer, doc_term_mat, vocab

    def fit(self, docs):
        lemmatized_docs, vectorizer, doc_term_mat, vocab = self.preprocess(docs)
        self.vectorizer = vectorizer
        # Create and fit the LDA model
        lda = LDA(n_components=self.n_topics, n_jobs=-1, max_iter=20, learning_offset=100, learning_decay=0.51)
        X = lda.fit_transform(doc_term_mat)
        topic_feature_sets = list(zip(docs, lemmatized_docs, X))
        self.topic_feature_sets = topic_feature_sets
        # Print the topics found by the LDA model
        topics = []
        for topic in lda.components_:
            topic = [e for e in topic.argsort() if vocab[e] not in stopwords]
            topic_tokens = [vocab[i] for i in topic[:-self.n_words - 1:-1]]
            topics.append(topic_tokens)
        self.model = lda
        self.topics = topics
        return X
        #return lda, topics, topic_feature_sets

    def transform(self, docs):
        return self.model.transform(self.vectorizer.transform(docs))

    def histplot(self, topic_idx):
        sns.distplot([e[2][topic_idx] for e in self.topic_feature_sets])

def find_topics(threshold=0.99, chunk_size=1000, n=243930):
    print("Loading relevance classifier...")
    relevance_classifier = build_classifier()
    print("Labeling first " + str(n) + " posts by relevance...")
    feature_sets = label_data(relevance_classifier, chunk_size=chunk_size, n=n)
    size = len(feature_sets)
    with open('generated_relevance_feature_sets.pkl', mode="wb") as f:
        pickle.dump(feature_sets, f)
    print("Filtering relevant posts by relevance threshold=" + str(threshold))
    feature_sets = [fset for fset in feature_sets if fset[1][1]>=threshold]
    print(str(len(feature_sets)) + " posts kept out of " + str(size))
    start = time.time()
    print("Extracting topics from the remaining " + str(len(feature_sets)) + " most relevant posts.")
    relevant_corpus = Corpus([e[0] for e in feature_sets], tokenize=True)

    miner = TopicMiner()
    X = miner.fit(relevant_corpus.docs)
    topic_feature_sets = list(zip(relevant_corpus.docs, X))

    with open('generated_topic_feature_sets.pkl', mode="wb") as f:
        pickle.dump(topic_feature_sets, f)
    print("saving topic model...")
    with open('lda_topic_model.pkl', mode="wb") as f:
        pickle.dump(miner, f)
    print("Took " + str(time.time()-start) + " seconds to extract topics.")
    return topic_feature_sets, TopicMiner


def extract_topics(corpus, n_components=200, n_topics=10, n_words=10):
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        ngram_range=(1,2),
        stop_words = ignore_tokens
    )

    docs = corpus.docs
    lemmatized_docs = [lemmatize(doc) for doc in docs]


    doc_term_mat = vectorizer.fit_transform(lemmatized_docs)
    vocab = vectorizer.get_feature_names()
    #pca = TruncatedSVD(n_components=n_components)
    #result = pca.fit_transform(doc_term_mat)

    # Create and fit the LDA model
    lda = LDA(n_components=n_topics, n_jobs=-1, max_iter=20, learning_offset=100, learning_decay=0.51)
    X = lda.fit_transform(doc_term_mat)
    topic_feature_sets = list(zip(docs, lemmatized_docs, X))
    # Print the topics found by the LDA model
    topics = []
    for topic in lda.components_:
        topic = [e for e in topic.argsort() if vocab[e] not in stopwords]
        topic_tokens = [vocab[i] for i in topic[:-n_words - 1:-1]]
        topics.append(topic_tokens)
    return lda, topics, topic_feature_sets


def find_relevant_topics(threshold=0.99, chunk_size=1000, n=243930):
    print("Loading relevance classifier...")
    relevance_classifier = build_classifier()
    print("Labeling first " + str(n) + " posts by relevance...")
    feature_sets = label_data(relevance_classifier, chunk_size=chunk_size, n=n)
    size = len(feature_sets)
    with open('generated_relevance_feature_sets.pkl', mode="wb") as f:
        pickle.dump(feature_sets, f)
    print("Filtering relevant posts by relevance threshold=" + str(threshold))
    feature_sets = [fset for fset in feature_sets if fset[1][1]>=threshold]
    print(str(len(feature_sets)) + " posts kept out of " + str(size))
    start = time.time()
    print("Extracting topics from the remaining " + str(len(feature_sets)) + " most relevant posts.")
    relevant_corpus = Corpus([e[0] for e in feature_sets], tokenize=True)
    lda, topics, topic_feature_sets = extract_topics(relevant_corpus)
    with open('generated_topic_feature_sets.pkl', mode="wb") as f:
        pickle.dump(topic_feature_sets, f)
    print("saving topic model...")
    with open('lda_topic_model.pkl', mode="wb") as f:
        pickle.dump(lda, f)
    print("Took " + str(time.time()-start) + " seconds to extract topics.")
    return lda, topics, topic_feature_sets, relevance_classifier

def load_lda_model():
    with open('lda_topic_model.pkl', mode="rb") as f:
        lda = pickle.load(f)
    with open("generated_topic_feature_sets.pkl", mode="rb") as f:
        feature_sets = pickle.load(f)
    return lda, feature_sets

def train_topic_classifier(test_size=0.9, n_labels=10):
    with open('generated_topic_feature_sets.pkl', mode="rb") as f:
        feature_sets = pickle.load(f)
        docs = [e[1] for e in feature_sets]
        labels = [e[2] for e in feature_sets]
        corpus = Corpus(docs, tokenize=False)
        print("Training w2v...")
        corpus.fit_w2v('w2v/w2v_topic.model')
        return corpus.build_classifier(docs, labels, path="topic_classifier.h5", test_size=test_size)
