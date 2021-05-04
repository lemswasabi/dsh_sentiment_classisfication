#!/usr/bin/env python3

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

parameters = \
    [ \
        {
            'clf': [MultinomialNB()],
            'alpha': [0.001, 0.1, 1, 10, 100]
        },

        {
            'clf': [SVC()],
            'C': [0.001, 0.1, 1, 10, 100, 10e5],
            'kernel': ['linear', 'poly', 'rbf'],
            'class_weight': ['balanced'],
            'probability': [True]
        },

        {
            'clf': [KNeighborsClassifier()],
            'n_neighbors': [*range(5, 100, 10)],
            'p': [1, 2]
        },

        {
            'clf': [RandomForestClassifier()],
            'n_estimators': [*range(10, 100, 10)],
            'criterion': ['gini','entropy'],
            'class_weight':['balanced', 'balanced_subsample', None]
        },

        {
            'clf': [DecisionTreeClassifier()],
            'criterion': ['gini','entropy'],
            'splitter': ['best','random'],
            'class_weight':['balanced', None]
        }

    ]

