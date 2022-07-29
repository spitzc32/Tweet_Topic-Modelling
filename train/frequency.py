import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk import  word_tokenize
from sklearn.metrics import accuracy_score, classification_report

from train.classifier import Cleaner 
from model.frequency import ClassModel


def train():
    ds_train = pd.read_csv("./data/disaster_tweets_en.csv")
    cleaner = Cleaner(
        ds_train=ds_train, 
        train_column="rawContent", 
        test_column="keyword",
        type="train"
    )
    X,y = cleaner.X_train, cleaner.y_train
    
    classifiers =ClassModel(
        X_train=X,
        Y_train=y,
    )
    classifiers_model = classifiers.fit()

    for classifier in classifiers_model:
        predictions = classifier[1].predict(cleaner.X_test)

        class_report = classification_report(cleaner.y_test, predictions)
        print('\n Accuracy: ', accuracy_score(cleaner.y_test, predictions))
        print('\nClassification Report for')
        print('======================================================')
        print('\n', class_report)

        classifiers.save(f"./saved_models/{classifier[0]}.pkl", classifier[1])
        
        