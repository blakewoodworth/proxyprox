import os
import json
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class AmazonSentiment:

    def __init__(self, max_features=1000, n_examples=50000, proxy_label_1=False, vectorizer=None):

        datafile = './data/amazon_train.ft.txt'
        if n_examples is not None:
            data = pd.read_csv(datafile, sep='\t', names=['text'], nrows=n_examples)
        else:
            data = pd.read_csv(datafile, sep='\t', names=['text'])
        data['label'] = data['text'].str.extract(r'__label__(\d)').astype(int) - 1
        data['review_text'] = data['text'].str.replace(r'__label__\d ', '', regex=True)
        data = data[['review_text', 'label']]

        stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
                "your", "yours", "yourself", "yourselves", "he", "him", "his", 
                "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
                "they", "them", "their", "theirs", "themselves", "what", "which", 
                "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
                "was", "were", "be", "been", "being", "have", "has", "had", "having", 
                "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
                "or", "because", "as", "until", "while", "of", "at", "by", "for", 
                "with", "about", "against", "between", "into", "through", "during", 
                "before", "after", "above", "below", "to", "from", "up", "down", "in", 
                "out", "on", "off", "over", "under", "again", "further", "then", 
                "once", "here", "there", "when", "where", "why", "how", "all", "any", 
                "both", "each", "few", "more", "most", "other", "some", "such", "no", 
                "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
                "t", "can", "will", "just", "don", "should", "now"]

        train_text, test_text, Y_train, Y_test = train_test_split(data['review_text'], data['label'], test_size=0.1, random_state=1)
        proxy_text, target_text, Y_proxy, Y_target = train_test_split(train_text, Y_train, test_size=0.111111111, random_state=1)

        if vectorizer is None:
            vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
            X_target = vectorizer.fit_transform(target_text)
        else:
            X_target = vectorizer.transform(target_text)
        X_proxy = vectorizer.transform(proxy_text)
        X_test = vectorizer.transform(test_text)

        self.vectorizer = vectorizer

        X_target = sp.sparse.hstack((X_target, np.ones((X_target.shape[0],1)))).toarray()
        X_proxy = sp.sparse.hstack((X_proxy, np.ones((X_proxy.shape[0],1)))).toarray()
        X_test = sp.sparse.hstack((X_test, np.ones((X_test.shape[0],1)))).toarray()

        Y_target = np.array(Y_target).astype(np.int32)
        if not proxy_label_1:
            Y_proxy = np.array(Y_proxy).astype(np.int32)
        else:
            Y_proxy = np.ones(Y_proxy.shape[0]).astype(np.int32)
        Y_test = np.array(Y_test).astype(np.int32)

        self.target = (X_target, Y_target)
        self.proxy = (X_proxy, Y_proxy)
        self.test = (X_test, Y_test)

        self.d = (X_target.shape[1],)
        self.n_classes = 2


class YelpSentiment:

    def __init__(self, max_features=1000, n_examples=50000, proxy_label_1=False, vectorizer=None):

        data = {}
        data['review_text'] = []
        data['label'] = []

        datafile = './data/yelp.json'
        with open(datafile, 'r') as f:
            for i,line in enumerate(f):
                if i >= n_examples:
                    break
                d = json.loads(line)
                data['review_text'].append(d['text'])
                data['label'].append(int(d['stars']))

        data['review_text'] = np.array(data['review_text'])
        data['label'] = np.array(data['label'])

        stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
                "your", "yours", "yourself", "yourselves", "he", "him", "his", 
                "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
                "they", "them", "their", "theirs", "themselves", "what", "which", 
                "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
                "was", "were", "be", "been", "being", "have", "has", "had", "having", 
                "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
                "or", "because", "as", "until", "while", "of", "at", "by", "for", 
                "with", "about", "against", "between", "into", "through", "during", 
                "before", "after", "above", "below", "to", "from", "up", "down", "in", 
                "out", "on", "off", "over", "under", "again", "further", "then", 
                "once", "here", "there", "when", "where", "why", "how", "all", "any", 
                "both", "each", "few", "more", "most", "other", "some", "such", "no", 
                "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
                "t", "can", "will", "just", "don", "should", "now"]

        train_text, test_text, Y_train, Y_test = train_test_split(data['review_text'], data['label'], test_size=0.1, random_state=1)
        proxy_text, target_text, Y_proxy, Y_target = train_test_split(train_text, Y_train, test_size=0.111111111, random_state=1)

        if vectorizer is None:
            vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
            X_target = vectorizer.fit_transform(target_text)
        else:
            X_target = vectorizer.transform(target_text)
        X_proxy = vectorizer.transform(proxy_text)
        X_test = vectorizer.transform(test_text)

        self.vectorizer = vectorizer

        X_target = sp.sparse.hstack((X_target, np.ones((X_target.shape[0],1)))).toarray()
        X_proxy = sp.sparse.hstack((X_proxy, np.ones((X_proxy.shape[0],1)))).toarray()
        X_test = sp.sparse.hstack((X_test, np.ones((X_test.shape[0],1)))).toarray()

        Y_target = np.array(Y_target).astype(np.int32) - 1
        if not proxy_label_1:
            Y_proxy = np.array(Y_proxy).astype(np.int32) - 1
        else:
            Y_proxy = np.ones(Y_proxy.shape[0]).astype(np.int32)
        Y_test = np.array(Y_test).astype(np.int32) - 1

        self.target = (X_target, Y_target)
        self.proxy = (X_proxy, Y_proxy)
        self.test = (X_test, Y_test)

        self.d = (X_target.shape[1], 5)
        self.n_classes = 5


class ShakespeareText:

    def __init__(self, max_features=1000, n_examples=50000, vectorizer=None):

        assert vectorizer is not None

        data = []
        datafile = './data/shakespeare.txt'
        with open(datafile, 'r') as f:
            current = ''
            for line in f.readlines():
                if len(data) >= n_examples:
                    break
                current += ' ' + line.strip()
                if len(current) >= 200:
                    data.append(current)
                    current = ''
                
        stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
                "your", "yours", "yourself", "yourselves", "he", "him", "his", 
                "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
                "they", "them", "their", "theirs", "themselves", "what", "which", 
                "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
                "was", "were", "be", "been", "being", "have", "has", "had", "having", 
                "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
                "or", "because", "as", "until", "while", "of", "at", "by", "for", 
                "with", "about", "against", "between", "into", "through", "during", 
                "before", "after", "above", "below", "to", "from", "up", "down", "in", 
                "out", "on", "off", "over", "under", "again", "further", "then", 
                "once", "here", "there", "when", "where", "why", "how", "all", "any", 
                "both", "each", "few", "more", "most", "other", "some", "such", "no", 
                "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
                "t", "can", "will", "just", "don", "should", "now"]

        X_proxy = vectorizer.transform(data)
        X_proxy = sp.sparse.hstack((X_proxy, np.ones((X_proxy.shape[0],1)))).toarray()
        Y_proxy = np.ones(X_proxy.shape[0]).astype(np.int32)

        self.proxy = (X_proxy, Y_proxy)

class AdultDataset:

    def __init__(self):

        if 'data' not in os.listdir():
            os.system('mkdir data')
        if 'adult.data' not in os.listdir('data'):
            os.system('curl -o data/adult.data http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
        colnames = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','country','salary']
        df = pd.read_csv('data/adult.data',sep=',',header=None,names=colnames)
        
        # featurize everything
        df['salary'] = df['salary'].map({' <=50K':1,' >50K':0}).astype(int)
        df['sex'] = df['sex'].map({' Male':1,' Female':0}).astype(int)

        df['country'] = df['country'].replace(' ?',np.nan)
        df['workclass'] = df['workclass'].replace(' ?',np.nan)
        df['occupation'] = df['occupation'].replace(' ?',np.nan)
        df.dropna(how='any',inplace=True)

        df.loc[df['country'] != ' United-States', 'country'] = 'Non-US'
        df.loc[df['country'] == ' United-States', 'country'] = 'US'
        df['country'] = df['country'].map({'US':1,'Non-US':0}).astype(int)

        df['marital-status'] = df['marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
        df['marital-status'] = df['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')
        df['marital-status'] = df['marital-status'].map({'Couple':0,'Single':1})

        # df['relationship1'] = df['relationship']
        df['relationship1'] = 1*(df['relationship'] == ' Unmarried')
        df['relationship2'] = 1*(df['relationship'] == ' Not-in-family')
        df['relationship3'] = 1*(df['relationship'] == ' Other-relative')
        df['relationship4'] = 1*(df['relationship'] == ' Wife')
        df['relationship'] = 1*(df['relationship'] == ' Own-child')

        df['race1'] = 1*(df['race'] == ' White')
        df['race2'] = 1*(df['race'] == ' Amer-Indian-Eskimo')
        df['race3'] = 1*(df['race'] == ' Asian-Pac-Islander')
        df['race'] = 1*(df['race'] == ' Black')

        def workclass_transform(x):
            if x['workclass'] in [' Federal-gov', ' Local-gov',' State-gov']: 
                return 'govt'
            elif x['workclass'] == ' Private':
                return 'private'
            elif x['workclass'] in [' Self-emp-inc', ' Self-emp-not-inc']: 
                return 'self_employed'
            else: 
                return 'without_pay'
        df['workclass'] = df.apply(workclass_transform, axis=1)
        df['workclass1'] = 1*(df['workclass'] == 'govt')
        df['workclass2'] = 1*(df['workclass'] == 'private')
        df['workclass'] = 1*(df['workclass'] == 'without_pay')
        df.drop(labels=['fnlwgt','education','occupation'],axis=1,inplace=True)

        feats = np.array(df.loc[:, df.columns != 'salary']).astype(np.float32)
        labels = np.array(df.loc[:, 'salary']).astype(np.int32)
        
        X_train, X_test, Y_train, Y_test = train_test_split(feats, labels, test_size=0.15, random_state=1)
        X_proxy, X_target, Y_proxy, Y_target = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)


        Y_proxy = np.ones(Y_proxy.shape[0])

        n_components = 14
        pca = PCA(n_components=n_components)
        X_target = pca.fit_transform(X_target)
        X_proxy = pca.transform(X_proxy)
        X_test = pca.transform(X_test)

        scaler = StandardScaler()
        X_target = scaler.fit_transform(X_target)
        X_proxy = scaler.transform(X_proxy)
        X_test = scaler.transform(X_test)

        X_target = np.concatenate([X_target, np.ones((X_target.shape[0],1))], axis=1)
        X_proxy = np.concatenate([X_proxy, np.ones((X_proxy.shape[0],1))], axis=1)
        X_test = np.concatenate([X_test, np.ones((X_test.shape[0],1))], axis=1)
        
        self.target = (X_target, Y_target)
        self.proxy = (X_proxy, Y_proxy)
        self.test = (X_test, Y_test)

        self.d = (n_components+1,)
        self.n_classes = 2






