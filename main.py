#!/usr/bin/env python3

from src.gridsearch import grid_search
from utils.utils import preprocess_dataset

def main():
    X, y = preprocess_dataset('data_text/trainset.txt')
    y_sentiment = y['sentiment']
    y_topic = y['topic']

    print('\n\nGrid search for sentiment classification:')

    results_sentiment = grid_search(X, y_sentiment)

    for result in results_sentiment:

        print("Classifier %s %0.3f (+/-%0.03f) for %r" % (result['classifier'], result['best_score'], result['std_test_score'] * 2, result['best_params']))

    print('\n\nGrid search for topic classification:')

    results_topic = grid_search(X, y_topic)

    for result in results_topic:

        print("Classifier %s %0.3f (+/-%0.03f) for %r" % (result['classifier'], result['best_score'], result['std_test_score'] * 2, result['best_params']))

if __name__ == '__main__':
    main()
