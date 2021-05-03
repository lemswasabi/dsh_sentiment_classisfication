#!/usr/bin/env python3

import nltk
import string
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

def preprocess_dataset(path, text_representation='tfid', feature_selection=None, labels='both'):
    """
    Preprocess dataset and return features and labels
    Args:
        path: path string to trainset.txt
        text_representation: representation of text, default tfid
        feature_selection: type of feature_selection, default None
        labels: return type of labels, default both
    Return:
        (processedText, labels): preprocessed features and labels (contains binarized 'sentiment' and/or label encoding of 'topic' and label 'topic_labels')
    """

    reviews = trainset_to_df(path)

    # Removing punctuations
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
    if text_representation == 'tfid':

        vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8)
        processedText = vectorizer.fit_transform(reviews['text']).toarray()

    # Get labels
    if labels == 'both':
        labels = reviews.drop('text', axis=1)
    elif labels == 'sentiment':
        labels = reviews['sentiment']
    else:
        labels = reviews[['topic', 'topic_labels']]

    return (processedText, labels)

if __name__ == '__main__':
    df = trainset_to_df('../data_text/trainset.txt')
    X, y = preprocess_dataset('../data_text/trainset.txt')
