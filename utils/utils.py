#!/usr/bin/env python3

import nltk
import string
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

def trainset_to_df(path):
    """
    trainset_to_df converts trainset.txt to a pandas dataframe
    Args:
        path: path string to trainset.txt
    Return:
        df: pandas dataframe of trainset.txt
    """

    with open(path, 'r') as f:
        lines = f.readlines()

    lines = [[line.split()[0], line.split()[1], line.split()[2], ' '.join(line.split()[3:])] for line in lines]
    df = pd.DataFrame(lines, columns=['topic', 'sentiment', 'id', 'text'])

    return df

def preprocess_dataset(path, remove_punctuation=True,
                       text_representation='tfidf', tfidf_max_features=None,
                       tfidf_min_df=7, tfidf_max_df=0.8,
                       feature_selection=None, labels='sentiment'):
    """
    Preprocess dataset and return features and labels
    Args:
        path: path string to trainset.txt
        remove_punctuation: remove punctuation flag, default True
        text_representation: representation of text, default tfidf
        tfidf_max_features: max_features for tfidf, default None
        tfidf_min_df: min_df for tfidf, default 7
        tfidf_max_df: max_df for tfidf, default 0.8
        feature_selection: type of feature_selection, default None
        labels: return type of labels, default sentiment
    Return:
        (features, y_labels): preprocessed features and labels ([contains
        binarized 'sentiment'] or [label encoding of 'topic' and label
        'topic_labels'])
    """

    reviews = trainset_to_df(path)

    # Removing punctuations
    if remove_punctuation:
        process_text = lambda review: ' '.join(review.translate(str.maketrans('', '', string.punctuation)).split())
        reviews['text'] = reviews['text'].apply(process_text)

    # Removing all stopwords
    stopword_list = stopwords.words('english')
    stopword_list += 'nt'
    reviews['text'] = reviews['text'].apply(lambda review: ' '.join([word for word in review.split() if word not in stopword_list]))

    # Drop id column
    reviews.drop('id', axis=1, inplace=True)

    # Binarize sentiment label
    reviews['sentiment'] = reviews['sentiment'].apply(lambda sentiment: 0 if sentiment == 'neg' else 1)

    # Label Encode topic label
    le = LabelEncoder()
    reviews['topic'] = le.fit_transform(reviews['topic'])
    reviews['topic_labels'] = le.inverse_transform(reviews['topic'])

    # Vectorize text with Tfid
    if text_representation == 'tfidf':

        vectorizer = TfidfVectorizer (max_features=tfidf_max_features, min_df=tfidf_min_df, max_df=tfidf_max_df)
        features = vectorizer.fit_transform(reviews['text']).toarray()

    # Get labels
    if labels == 'sentiment':
        y_labels = reviews['sentiment']
    else:
        y_labels = reviews[['topic', 'topic_labels']]

    # Feature selection
    if feature_selection == 'variance_threshold':

        from sklearn.feature_selection import VarianceThreshold
        var_thres = VarianceThreshold(threshold=np.var(features))
        features = var_thres.fit_transform(features)

    elif feature_selection == 'chi_square_test':

        from sklearn.feature_selection import SelectKBest, chi2
        chi2 = SelectKBest(chi2, k=features.shape[1])
        if labels == 'topic':
            features = chi2.fit_transform(features, y_labels.iloc[:, 0])
        else:
            features = chi2.fit_transform(features, y_labels)

    return (features, y_labels)

if __name__ == '__main__':
    df = trainset_to_df('../data_text/trainset.txt')
    X, y = preprocess_dataset('../data_text/trainset.txt', feature_selection='chi_square_test', labels='sentiment')

    print(X, y)
