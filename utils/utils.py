#!/usr/bin/env python3

import pandas as pd

def trainset_to_df(path, representation='bag_of_words'):
    """
    trainset_to_df converts trainset.txt to a pandas dataframe
    Args:
        path: path string to trainset.txt
        representation: represenation of text, defaul: bag of words
    Return:
        df: pandas dataframe of trainset.txt
    """

    with open(path, 'r') as f:
        lines = f.readlines()

    lines = [[line.split()[0], line.split()[1], line.split()[2], ' '.join(line.split()[3:])] for line in lines]
    df = pd.DataFrame(lines, columns=['topic', 'sentiment', 'id', 'text'])

    return df

if __name__ == '__main__':
    df = trainset_to_df()
