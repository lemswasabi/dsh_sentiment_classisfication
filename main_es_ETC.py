#!/usr/bin/env python3

import pickle

from src.gridsearch import grid_search
from utils.utils import preprocess_dataset

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC
import xgboost as xgb

parameters = \
    [ \

        {
            'clf': [ExtraTreesClassifier()],
            'n_estimators': [10, 30, 50, 70, 90],
            'criterion': ['gini','entropy'],
            'class_weight':['balanced', None],
            'max_samples': [100, 200, 300, 400, 500]
        },
    ]

def main():
    X, y = preprocess_dataset('data_text/trainset.txt', remove_punctuation=False , feature_selection='chi_square_test')

    results_sentiment = grid_search(X, y, parameters)

    with open('results/results_sentiment_with_punctuation_es_ETC.pickle', 'wb') as handle:
        pickle.dump(results_sentiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X, y = preprocess_dataset('data_text/trainset.txt', remove_punctuation=False , feature_selection='chi_square_test', labels='topic')

    results_topic = grid_search(X, y['topic'], parameters)

    with open('results/results_topic_with_punctuation_es_ETC.pickle', 'wb') as handle:
        pickle.dump(results_topic, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
