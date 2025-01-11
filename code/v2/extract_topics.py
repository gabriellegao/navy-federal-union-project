import numpy, pandas

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans

from nltk.corpus import stopwords

def load_messages():
    return pandas.read_csv('../../data/dataset.csv').NormalizedMessage.astype(str)

if __name__=='__main__':
    docs = load_messages()
    stop_words = list(stopwords.words('english'))
    vec = TfidfVectorizer(stop_words=stop_words)
    doc_term_mat = vec.fit_transform(docs)
    print(doc_term_mat)
