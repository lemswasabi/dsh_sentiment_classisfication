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
            'clf': [AdaBoostClassifier()],
            'base_estimator': [MultinomialNB(alpha=10),
                                DecisionTreeClassifier(splitter='random'),
                                SVC(kernel='rbf', C=10, class_weight='balanced', probability=True),
                                LogisticRegression(solver='liblinear'),
                                KNeighborsClassifier(n_neighbors=95)],
            'n_estimators': [50, 60, 70, 80, 100],
            'learning_rate': [0.3, 0.5, 0.8, 1.0, 1.5]
        },
    ]

def main():
    X, y = preprocess_dataset('data_text/trainset.txt', remove_punctuation=False , feature_selection='chi_square_test')

    results_sentiment = grid_search(X, y, parameters)

    with open('results/results_sentiment_with_punctuation_es_ABC.pickle', 'wb') as handle:
        pickle.dump(results_sentiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X, y = preprocess_dataset('data_text/trainset.txt', remove_punctuation=False , feature_selection='chi_square_test', labels='topic')

    results_topic = grid_search(X, y['topic'], parameters)

    with open('results/results_topic_with_punctuation_es_ABC.pickle', 'wb') as handle:
        pickle.dump(results_topic, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
