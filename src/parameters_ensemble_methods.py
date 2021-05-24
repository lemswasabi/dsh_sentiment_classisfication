#!/usr/bin/env python3

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

    # BAGGING

        {
            'clf': [BaggingClassifier()],
            'base_estimator': [MultinomialNB(alpha=10),
                                DecisionTreeClassifier(splitter='random'),
                                SVC(kernel='rbf', C=10, class_weight='balanced', probability=True),
                                LogisticRegression(solver='liblinear'),
                                KNeighborsClassifier(n_neighbors=95)],
            'n_estimators': [50, 60, 70, 80, 100],
            'max_samples': [100, 200, 300, 400, 500]
        },

        {
            'clf': [ExtraTreesClassifier()],
            'n_estimators': [10, 30, 50, 70, 90],
            'criterion': ['gini','entropy'],
            'class_weight':['balanced', None],
            'max_samples': [100, 200, 300, 400, 500]
        },

    # BOOSTING

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

        {
            'clf': [GradientBoostingClassifier()],
            'n_estimators': [70, 80, 100, 120, 130],
            'learning_rate': [0.1, 0.2, 0.3, 0.5, 0.8]
        },

        {
            'clf': [xgb.XGBClassifier()],
            'n_estimators': [70, 80, 100, 120, 130],
            'learning_rate': [0.1, 0.2, 0.3, 0.5, 0.8]
        },

    # STACKING

        {
            'clf': [StackingClassifier(estimators=[
                ('mnb', MultinomialNB(alpha=10)),
                ('svc', SVC(kernel='rbf', C=10, class_weight='balanced', probability=True)),
                ('lr', LogisticRegression(solver='liblinear')),
                ('knn', KNeighborsClassifier(n_neighbors=95))
            ])],
            'final_estimator': [MultinomialNB(alpha=10),
                                DecisionTreeClassifier(splitter='random'),
                                SVC(kernel='rbf', C=10, class_weight='balanced', probability=True),
                                LogisticRegression(solver='liblinear'),
                                KNeighborsClassifier(n_neighbors=95)],
        }
    ]
