import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans

def extract_topics_lda(cvec, docs, number_topics = 10, number_words = 10):
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(cvec.fit_transform(docs))
    # Print the topics found by the LDA model
    words = cvec.get_feature_names()
    topics = []
    for topic in lda.components_:
        topics.append([words[i] for i in topic.argsort()[:-number_words - 1:-1]])
    return topics


def word2vec(docs):
    # Trains and returns w2v model on docs
    pass

def compute_clusters(doc_term_mat):
    kmeans = KMeans(n_clusters=10, random_state=0).fit(vectorized_docs)
    return list(zip(doc_term_mat, kmeans.labels_))
