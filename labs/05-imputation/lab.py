# lab.py


import os
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------



def first_round():
    """
    :return: list with two values as described in the notebook
    >>> out = first_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] in ['NR', 'R']
    True
    """
    return [0.15, 'NR' ]


def second_round():
    """
    :return: list with three values as described in the notebook
    >>> out = second_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] in ['NR', 'R']
    True
    >>> out[2] in ['ND', 'D']
    True
    """
    return [0.0344518152440148, 'R', 'D']


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def verify_child(heights):
    """
    Returns a Series of p-values assessing the missingness
    of child-height columns on father height.
    
    :Example:
    >>> heights_fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(heights_fp)
    >>> out = verify_child(heights)
    >>> out['child_50'] < out['child_95']
    True
    >>> out['child_5'] > out['child_50']
    True
    """
    indeces = heights.columns
    #ser = pd.Series()
    names = []
    k_stats = []
    for n in indeces:
        if n.find('child_') != -1:
            heights[n + ' missing'] = heights[n].isna()
            
            names.append(n)
            k_stats.append(stats.ks_2samp(
            heights.groupby(n + ' missing')[n].get_group(True),
            heights.groupby(n + ' missing')[n].get_group(False)
            ).pvalue)
            
    #heights

    ser = pd.Series(index = names, data = k_stats)
    return ser


def missing_data_amounts():
    """
    Returns a list of multiple choice answers.
    :Example:
    >>> set(missing_data_amounts()) <= set(range(1, 6))
    True
    """
    return [2,4,5]


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cond_single_imputation(new_heights):
    """
    cond_single_imputation takes in a DataFrame with columns 
    father and child (with missing values in child) and imputes 
    single-valued mean imputation of the child column, 
    conditional on father. Your function should return a Series.

    :Example:
    >>> heights_fp = os.path.join('data', 'missing_heights.csv')
    >>> new_heights = pd.read_csv(heights_fp)[['father', 'child_50']]
    >>> new_heights = new_heights.rename(columns={'child_50': 'child'})
    >>> out = cond_single_imputation(new_heights)
    >>> out.isna().sum() == 0
    True
    >>> (new_heights['child'].std() - out.std()) > 0.5
    True
    """
    new_heights['cuts'] = pd.qcut(new_heights['father'],4)
    ser = new_heights.groupby('cuts')['child'].transform(mean_impute)#[:50]
    return ser



def mean_impute(ser):
    return ser.fillna(ser.mean())
# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    quantitative_distribution that takes in a Series and an integer 
    N > 0, and returns an array of N samples from the distribution of 
    values of the Series as described in the question.
    :Example:
    >>> heights_fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(heights_fp)
    >>> child = heights['child_50']
    >>> out = quantitative_distribution(child, 100)
    >>> out.min() >= 56
    True
    >>> out.max() <= 79
    True
    >>> np.isclose(out.mean(), child.mean(), atol=1)
    True
    >>> np.isclose(out.std(), 3.5, atol=0.65)
    True
    """
    demo = child.copy()
    demo = demo.dropna()
    probs = []
    
    for n in np.arange(N):
        hist, edges = np.histogram(demo, bins=10)
        #hist
        #edges
        index = np.random.choice(a=[0,1,2,3,4,5,6,7,8,9], p=hist/hist.sum())
        probs.append(np.random.uniform(edges[index], edges[index+1]))
        
    return np.array(probs)


def impute_height_quant(child):
    """
    impute_height_quant takes in a Series of child heights 
    with missing values and imputes them using the scheme in
    the question.
    :Example:
    >>> heights_fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(heights_fp)
    >>> child = heights['child_50']
    >>> out = impute_height_quant(child)
    >>> out.isna().sum() == 0
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.5)
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.2)
    True
    >>> np.isclose(out.std(), child.std(), atol=0.15)
    True
    """
    ser = child.copy()
    missing = ser.isna().sum()
    #missing
    out = quantitative_distribution(ser,missing)
    #out
    ser.update(pd.Series(out, index = ser.index[ser.isna()]))
    return ser


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def answers():
    """
    Returns two lists with your answers
    :return: Two lists: one with your answers to multiple choice questions
    and the second list has 6 websites that satisfy given requirements.
    >>> mc_answers, websites = answers()
    >>> len(mc_answers)
    4
    >>> len(websites)
    6
    """
    return [1,2,2,1], ['https://www.jpg.store/robots.txt','https://companiesmarketcap.com/robots.txt','https://www.johnma.design/robots.txt','https://www.netflix.com/robots.txt', 'https://www.instagram.com/robots.txt', 'https://www.facebook.com/robots.txt' ]
