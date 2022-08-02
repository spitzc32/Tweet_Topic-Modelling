import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle
from sklearn.metrics import accuracy_score, classification_report

from train.classifier import Cleaner 
from model.frequency import ClassModel


def train():
    ds_train = pd.read_csv("./data/disaster_tweets_en.csv")
    concat_ds = pd.read_csv("./data/disaster_tweets_en_test.csv")
    concat_tl = pd.read_csv("./data/disaster_tweets_tl.csv")

    concated_train = pd.concat([ds_train, concat_ds])
    cleaner = Cleaner(
        ds_train=concated_train, 
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

def test_accuracy():
    ds_train = pd.read_csv("./data/disaster_tweets_en.csv")
    concat_tl = pd.read_csv("./data/disaster_tweets_tl.csv")

    concated_train = pd.concat([ds_train, concat_tl])
    cleaner = Cleaner(
        ds_train=concated_train, 
        train_column="rawContent", 
        test_column="keyword",
        type="train"
    )
    X,y = cleaner.X_test, cleaner.y_test

    with open("./saved_models/LogisticRegressionFreq.pkl", 'rb') as file:
        pickle_model = pickle.load(file)

    predictions = pickle_model.predict(X)

    class_report = classification_report(y, predictions)
    print('\n Accuracy: ', accuracy_score(y, predictions))
    print('\nClassification Report for')
    print('======================================================')
    print('\n', class_report)

        
        