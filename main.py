#!/usr/bin/env python3

import logging

from src.gridsearch import grid_search
from utils.utils import preprocess_dataset

output_file = 'main.out'

def main():
    X, y = preprocess_dataset('data_text/trainset.txt')
    y_sentiment = y['sentiment']
    y_topic = y['topic']

    with open(output_file, 'w') as f:
        print('Grid search for sentiment classification:', file=f)

    results_sentiment = grid_search(X, y_sentiment)

    for result in results_sentiment:

        with open(output_file, 'a') as f:
            print("Classifier %s %0.3f (+/-%0.03f) for %r" % (result['classifier'], result['best_score'], result['std_test_score'] * 2, result['best_params']), file=f)

    with open(output_file, 'a') as f:
        print('\n\nGrid search for topic classification:', file=f)

    results_topic = grid_search(X, y_topic)

    for result in results_topic:

        with open(output_file, 'a') as f:
            print("Classifier %s %0.3f (+/-%0.03f) for %r" % (result['classifier'], result['best_score'], result['std_test_score'] * 2, result['best_params']), file=f)

if __name__ == '__main__':
    main()
