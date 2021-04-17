#!/usr/bin/env python3

import pandas as pd

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
        lines = [line.rstrip('\n').strip() for line in lines]

    lines = [[line.split()[0], line.split()[1], line.split()[2], line.split()[3:]] for line in lines]
    df = pd.DataFrame(lines, columns=['topic', 'sentiment', 'id', 'text'])

    return df

if __name__ == '__main__':
    df = trainset_to_df()
