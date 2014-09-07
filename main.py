# coding: utf-8

__author__ = 'sergeygolubev'

"""
Benchmarks for the Avito fraud detection competition
"""
import nltk.corpus
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from nltk import SnowballStemmer
import logging
from sklearn.externals import joblib
from pandas import read_csv, DataFrame
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import binarize
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier

from scipy.sparse import *
from scipy import *

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import AdaBoostClassifier

import time
import sys

import cPickle as pickle

dataFolder = "/Users/sergeygolubev/Documents/kaggle/avito/avitofixed/data/"
matrixfile = "data/matrixunigrbigramm_categ"
matrixcountsfile = "data/matrixcounts"
targetfile = "data/target.p"
counterfile = "data/counter.p"
tfidfcounterfile = "data/tfidfcounter.p"
classifierfile = "data/classifier.p"

matrixfiletest = "data/matrixunigrbigramm_categtest"
testitemidsfile = "data/tesxitemids.p"


rus_stopwords = frozenset(word.decode('utf-8') for word in nltk.corpus.stopwords.words("russian") if word!=u"не")
stemmer = SnowballStemmer('russian')
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)



def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed


def memoize(f):
    """ Decorator for caching."""
    class memodict(dict):
        __slots__ = ()
        def __missing__(self, key):
            self[key] = ret = f(key)
            return ret
    return memodict().__getitem__

@memoize
def correctWord (w):
    """ Corrects word by replacing characters with written similarly depending on which language the word.
        Fraudsters use this technique to avoid detection by anti-fraud algorithms."""

    if len(re.findall(ur"[а-я]",w))>len(re.findall(ur"[a-z]",w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)

@memoize
def stemmingWord (w):
    """ Stemming word."""
    return stemmer.stem(correctWord(w))

def getWords(text, stemmRequired = False, correctWordRequired = False):
    """ Splits the text into words, discards stop words and applies stemmer.
    Parameters
    ----------
    text : str - initial string
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required
    """
    cleanText = re.sub(u'[^a-zа-я]', ' ', text.lower())
    if correctWordRequired:
        words = [correctWord(w) if not stemmRequired or re.search("[0-9a-z]", w) else stemmingWord(correctWord(w)) for w in cleanText.split()]
    else:
        words = [w if not stemmRequired or re.search("[0-9a-z]", w) else stemmingWord(w) for w in cleanText.split()]

    return ' '.join(words)


class SimpleTokenizer(object):
    def __init__(self, stemmRequired = False, correctWordRequired = False):
        # self.stemmer = PorterStemmer()
        self.stemmRequired = stemmRequired
        self.correctWordRequired = correctWordRequired

    def __call__(self, text):
        return getWords(text, self.stemmRequired, self.correctWordRequired)


@timeit
def save_sparse_matrix(filename, matrix):
    matrix_coo=matrix.tocoo()
    row=matrix_coo.row
    col=matrix_coo.col
    data=matrix_coo.data
    shape=matrix_coo.shape
    np.savez(filename,row=row,col=col,data=data,shape=shape)

@timeit
def load_sparse_matrix(filename):
    y=np.load(filename)
    matrix= coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    matrix = matrix.tocsr()
    return matrix

@timeit
def data_load(filename, type="train"):

    data = read_csv(filename, sep='\t')
    print data.info()

    if type=="train":
        data = data.drop(['itemid','is_proved', 'close_hours'], axis=1)

    data.title[pd.isnull(data.title)] = " "
    data.description[pd.isnull(data.description)] = " "
    data.category[pd.isnull(data.category)] = " "
    data.subcategory[pd.isnull(data.subcategory)] = " "
    data.attrs[pd.isnull(data.attrs)] = " "

    data = data.dropna()
    print data.info()
    data.text = data.title + ' ' + data.description + ' ' + data.category + ' ' + data.subcategory + ' ' + data.attrs

    data.price[pd.isnull(data.price)] = 0
    data.phones_cnt[pd.isnull(data.phones_cnt)] = 0
    data.emails_cnt[pd.isnull(data.emails_cnt)] = 0
    data.urls_cnt[pd.isnull(data.urls_cnt)] = 0

    data.price = data.price/data.price.max()
    data.phones_cnt = data.phones_cnt/data.phones_cnt.max()
    data.emails_cnt = data.emails_cnt/data.emails_cnt.max()
    data.urls_cnt = data.urls_cnt/data.urls_cnt.max()
    # print data.phones_cnt

    if type=="train":
        target = data.is_blocked
        return data, target
    else:
        return data, data.itemid

@timeit
def pickle_save(data, filename):
    pickle.dump( data, open( filename, "wb" ), protocol=2)

@timeit
def pickle_load(filename):
    return pickle.load( open( filename, "rb" ) )

@timeit
def joblib_save(data, filename):
    joblib.dump( data, filename)

@timeit
def joblib_load(filename):
    return joblib.load( filename )


@timeit
def features_extraction(data, stemmRequired=True, correctWordRequired=True, ngram_range=(1,2), binary=True, stop_words=rus_stopwords):

    counter = CountVectorizer(preprocessor=SimpleTokenizer(stemmRequired=stemmRequired, correctWordRequired=correctWordRequired) ,ngram_range=ngram_range, binary=binary, stop_words=stop_words)
    features = counter.fit_transform(data)
    return features, counter

@timeit
def features_frequency(data, idf=False):

    tfidf_transformer = TfidfTransformer(use_idf=idf)
    features = tfidf_transformer.fit_transform(data)
    return features, tfidf_transformer


@timeit
def features_selection(data, target, num=500):

    featureselector = SelectKBest(score_func=chi2, k=num)

    features = featureselector.fit_transform(data, target)

    return features

@timeit
def sgdc_classification(data, target, kfold, alpha=1e-4, numiter=5, cvval=False):

    clf = SGDClassifier(    loss="log",
                            penalty="l2",
                            alpha=alpha,
                            class_weight='auto',
                            n_iter=numiter)

    clf.fit(data,target)
    predicted_scores = clf.predict(data)
    print np.mean(predicted_scores == target)

    if cvval:
        scores = cross_validation.cross_val_score(clf, data, target, cv = kfold)
        print sum(scores)/kfold

    return clf


def adaboost_calssification(data, target, alpha=1e-4):

    clf_base = KNeighborsClassifier()

    clf = AdaBoostClassifier(base_estimator=clf_base)

    clf.fit(data,target)
    predicted_scores = clf.predict(data)
    print np.mean(predicted_scores == target)



def csr_vappend(a,b):

    a = sp.vstack([a,b])
    print a.shape
    return a

def main():
    """ Generates features and fits classifier. """

    data, target = data_load('data/avito_train.tsv')

    counts = csr_vappend(sp.csr_matrix(data.price.values), sp.csr_matrix(data.emails_cnt.values))
    counts = csr_vappend(counts, sp.csr_matrix(data.phones_cnt.values))
    counts = csr_vappend(counts, sp.csr_matrix(data.urls_cnt.values))
    save_sparse_matrix(matrixcountsfile, counts.transpose())

    counts = load_sparse_matrix(matrixcountsfile+".npz")
    pickle_save(target, targetfile)

    features, counter = features_extraction(data.text, binary=False, stemmRequired=True, correctWordRequired=True)
    pickle_save(counter, counterfile)

    target = pickle_load(targetfile)
    features = load_sparse_matrix(matrixfile+".npz")
    # binarize(features, copy=False)
    # features = features_selection(features, np.array(target))

    features, tfidf_transformer = features_frequency(features, idf=True)
    pickle_save(tfidf_transformer, tfidfcounterfile)

    features = csr_vappend(features.transpose(), counts.transpose()).transpose()

    kfold = 5

    logging.info("Start classification...")

    # for alpha in 10.0**-np.arange(7,10):
    # for alpha in alphas:
    #     print alpha
    #     sgdc_classification(features, target, kfold, alpha=alpha, numiter=5, cvval=True)
    #
    # sys.exit()

    clf = sgdc_classification(features, target, kfold, alpha=7e-9, numiter=100)
    pickle_save(clf, classifierfile)

    counter = pickle_load(counterfile)
    tfidf_transformer = pickle_load(tfidfcounterfile)
    clf = pickle_load(classifierfile)
    data, testItemIds = data_load('data/avito_test.tsv', type="test")
    pickle_save(testItemIds, testitemidsfile)

    counts = csr_vappend(sp.csr_matrix(data.price.values), sp.csr_matrix(data.emails_cnt.values))
    counts = csr_vappend(counts, sp.csr_matrix(data.phones_cnt.values))
    counts = csr_vappend(counts, sp.csr_matrix(data.urls_cnt.values))

    testFeatures = load_sparse_matrix(matrixfiletest+".npz")
    testFeatures = counter.transform(data.text)
    save_sparse_matrix(matrixfiletest, testFeatures)

    # binarize(testFeatures, copy=False)
    testFeatures = tfidf_transformer.transform(testFeatures)

    testFeatures = csr_vappend(testFeatures.transpose(), counts).transpose()

    logging.info("Start prediction...")
    predicted_scores = clf.predict_proba(testFeatures).T[1]

    testItemIds = pickle_load(testitemidsfile)

    logging.info("Write results...")
    output_file = "avito_starter_solution.csv"
    logging.info("Writing submission to %s" % output_file)
    f = open(os.path.join(dataFolder,output_file), "w")
    f.write("id\n")

    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
        f.write("%d\n" % (item_id))

    f.close()
    logging.info("Done.")


if __name__=="__main__":
    main()