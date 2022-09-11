# discussion.py


import os
import numpy as np
import pandas as pd
import requests
import time
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def duplicate_words(s):
    """
    Provide a list of all words that are duplicates in an input sentence.
    Assume that the sentences are lower case.

    :Example:
    >>> duplicate_words('let us plan for a horror movie movie this weekend')
    ['movie']
    >>> duplicate_words('I like surfing')
    []
    >>> duplicate_words('the class class is good but the tests tests tests are hard')
    ['class', 'tests']
    """
    found = re.findall(r'\b(\w+)(\s+\1\b)+', s)
    simple = []
    for n in found:
        simple.append(n[0])
    return simple


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def laptop_details(df):
    """
    Given a df with product description - Return df with added columns of 
    processor (i3, i5), generation (9th Gen, 10th Gen), 
    storage (512 GB SSD, 1 TB HDD), display_inch (15.6 inch, 14 inch)

    :Example:
    >>> df = pd.read_csv('data/laptop_details.csv')
    >>> new_df = laptop_details(df)
    >>> new_df.shape
    (21, 5)
    >>> new_df['processor'].nunique()
    3
    """
    df['processor'] = df['laptop_description'].str.extract('Intel Core (i\d{1}) Processor')
    df
    df['generation'] = df['laptop_description'].str.extract('\(\s*([12345678910]{1,2}[a-z]{1,2}\s+Gen)\s*')
    df
    df['storage'] = df['laptop_description'].str.extract('(\d+ \w+ [HDS]{3})')
    df
    df['display_inch'] = df['laptop_description'].str.extract('([0-9]{0,2}[.]{0,2}[0-9]{0,2}\sinch)')
    return df


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def bigram_counts(review_df, column='reviewText'):
    """
    Given a DataFrame like `review_df`, return a Series with bi-gram counts sorted in descending order. 
    The index of the series should be a tuple of bi-grams 
    and the value should indicate the count of times that bi-gram appears in the whole corpus.

    :Example:
    >>> out_bigrams_text = bigram_counts(pd.read_csv('data/musical_instruments_reviews.csv'), 'reviewText')
    >>> isinstance(out_bigrams_text, pd.Series)
    True
    >>> out_bigrams_text.shape == (8470,)
    True
    >>> out_bigrams_text.index[0] == ('for', 'the')
    True
    """
    reviews = review_df[column].str.lower().str.replace('[^\w\s]','', regex=True)
    reviews = reviews.tolist()

    unigrams = []
    for review in reviews:
        words = review.split()
        zipy = zip(words[0::2], words[1::2])
        zipy2 = zip(words[1::2], words[2::2])
        unigrams.extend(zipy)
        unigrams.extend(zipy2)
    # print(unigrams)
    unigrams

    return pd.Series(unigrams).value_counts()
