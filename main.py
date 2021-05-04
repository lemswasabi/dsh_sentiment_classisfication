#!/usr/bin/env python3

import pickle

from src.gridsearch import grid_search
from utils.utils import preprocess_dataset

output_file = 'main.out'

def main():
    X, y = preprocess_dataset('data_text/trainset.txt')
    y_sentiment = y['sentiment']
    y_topic = y['topic']

    results_sentiment = grid_search(X, y_sentiment)

    with open('results/results_sentiment.pickle', 'wb') as handle:
        pickle.dump(results_sentiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # for result in results_sentiment:

    #     with open(output_file, 'a') as f:
    #         print("Classifier %s %0.3f (+/-%0.03f) for %r" % (result['classifier'], result['best_score'], result['std_test_score'] * 2, result['best_params']), file=f)

    results_topic = grid_search(X, y_topic)

    with open('results/results_topic.pickle', 'wb') as handle:
        pickle.dump(results_topic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # for result in results_topic:

    #     with open(output_file, 'a') as f:
    #         print("Classifier %s %0.3f (+/-%0.03f) for %r" % (result['classifier'], result['best_score'], result['std_test_score'] * 2, result['best_params']), file=f)

if __name__ == '__main__':
    main()
