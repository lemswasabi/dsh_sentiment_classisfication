#!/usr/bin/env python3

import pickle

from src.gridsearch import grid_search
from src.parameters import parameters
from utils.utils import preprocess_dataset

output_file = 'main.out'

def main():
    """
    A grid search framework to be used in a HPC environment
    """
    X, y = preprocess_dataset('data_text/trainset.txt', remove_punctuation=False , feature_selection='chi_square_test')

    results_sentiment = grid_search(X, y, parameters)

    with open('results/results_sentiment_with_punctuation.pickle', 'wb') as handle:
        pickle.dump(results_sentiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X, y = preprocess_dataset('data_text/trainset.txt', remove_punctuation=False , feature_selection='chi_square_test', labels='topic')

    results_topic = grid_search(X, y['topic'], parameters)

    with open('results/results_topic_with_punctuation.pickle', 'wb') as handle:
        pickle.dump(results_topic, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
