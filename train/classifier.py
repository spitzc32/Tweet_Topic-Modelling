import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from model.classifier import VoteClass
import pickle

class Cleaner():
    def __init__(self, ds_train, train_column, test_column, type) -> None:
        self.ds_train= ds_train
        self.train_column = train_column
        self.test_column = test_column
        self.type = type

        # Perform initial series of drops
        self.ds_train=self.ds_train.drop(['location'],axis=1)
        if "keyword" in self.ds_train:
            bool_series_keyword = pd.isnull(self.ds_train['keyword']) 
            
            #dropping missing 'keyword' records from train data set
            self.ds_train=self.ds_train.drop(self.ds_train[bool_series_keyword].index,axis=0)
            #Resetting the index after droping the missing records
            self.ds_train=self.ds_train.reset_index(drop=True)

        if "rawContent" in self.ds_train:
            # Drop non unique column
            self.ds_train = self.ds_train.sort_values('keyword').drop_duplicates('rawContent', keep='last')
            print(len(self.ds_train))
            # Filter Relation
            #.filter_no_relation(self.fit_testset(self.ds_train[self.train_column]))
        
       

        self.X = self.fit_testset(self.ds_train[self.train_column])
        self.y = self.ds_train[self.test_column]
        self.X_train , self.X_test ,self.y_train , self.y_test = train_test_split(self.X,self.y,test_size=0.1, random_state=10)
    
    def generate_corpus(self, train):
        
        texts = [txt.lower() for txt in train]
        #Remove urls
        texts = [re.sub(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+", "", txt) for txt in texts]
        #remove mentions
        texts = [re.sub(r'@\w+([-.]\w+)*',"", txt) for txt in texts]
        #Remove unwanted words
        texts = [re.sub("[^a-zA-Z]", ' ', txt)  for txt in texts]
        texts = [re.sub(r'&\w+([-.]\w+)*', '', text) for text in texts]
        texts = np.array(texts)
        stemmer = nltk.stem.PorterStemmer()
        for i in range(self.ds_train.shape[0]):
            words = nltk.word_tokenize(texts[i])
            sentence = ' '.join([stemmer.stem(word) for word in words if word not in stopwords.words('english')])
            texts[i] = sentence
        return texts
    
    def encode_output(self):
        label_encoder = LabelEncoder()
        y = np.array(self.ds_train[self.test_column])
        y = label_encoder.fit_transform(y)

        return y
    
    def filter_no_relation(self, train):
        with open("./saved_models/SVMClassifier.pkl", 'rb') as file:
            pickle_model = pickle.load(file)

        predictions = pickle_model.predict(train)

        self.ds_train["vote_pred"] = predictions
        self.ds_train = self.ds_train[self.ds_train["vote_pred"] != 0]

    
    def generate_Xy_vals(self, train):
        # Creating the Bag of Words model
        corpus = self.generate_corpus(train)
        td = TfidfVectorizer(max_features = 2000, stop_words=stopwords.words('english'), min_df=5, max_df=0.7, norm="l2")
        X = td.fit_transform(corpus).toarray()
        y = self.ds_test
        return X,y 
    
    def fit_testset(self, train):
        corpus = self.generate_corpus(train)
        td = TfidfVectorizer(max_features = 2000, stop_words=stopwords.words('english'), min_df=5, max_df=0.7, norm="l2")
        X = td.fit_transform(corpus).toarray()

        return X


def train():
    ds_train = pd.read_csv("./data/disaster_train.csv")
    cleaner = Cleaner(
        ds_train=ds_train,
        train_column="text", 
        test_column="target",
        type="train"
    )
    
    X,y = cleaner.X_train, cleaner.y_train
    
    vote_classifier = VoteClass(
        X_train=X,
        Y_train=y
    )
    vote_classifier.fit()

    predictions = vote_classifier.model.predict(cleaner.X_test)
    
    class_report = classification_report(cleaner.y_test, predictions)
    print('\n Accuracy: ', accuracy_score(cleaner.y_test, predictions))
    print('\nClassification Report for')
    print('======================================================')
    print('\n', class_report)

    vote_classifier.save("./saved_models/SGDClassifier.pkl")