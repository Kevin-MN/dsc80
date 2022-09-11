# lab.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer
import itertools
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def best_transformation():
    """
    Returns an integer corresponding to the correct option.

    :Example:
    >>> best_transformation() in [1, 2, 3, 4]
    True
    """
    return 1


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def create_ordinal(df):
    """
    create_ordinal takes in diamonds and returns a DataFrame of ordinal
    features with names ordinal_<col> where <col> is the original
    categorical column name.

    :Example:
    >>> diamonds = pd.read_csv(os.path.join('data', 'diamonds.csv'))
    >>> out = create_ordinal(diamonds)
    >>> set(out.columns) == {'ordinal_cut', 'ordinal_clarity', 'ordinal_color'}
    True
    >>> np.unique(out['ordinal_cut']).tolist() == [0, 1, 2, 3, 4]
    True
    """
    new_df = pd.DataFrame(columns = ['ordinal_cut','ordinal_color', 'ordinal_clarity'])
    new_df['ordinal_cut'] = df['cut'].apply(trans_cut)
    new_df['ordinal_color'] = df['color'].apply(trans_color)
    new_df['ordinal_clarity'] = df['clarity'].apply(trans_clarity)
    return new_df



def trans_cut(label):
    if label == 'Fair':
        return 0
    elif label == 'Good':
        return 1
    elif label == 'Very Good':
        return 2
    elif label == 'Premium':
        return 3
    elif label == 'Ideal':
        return 4
    

def trans_color(label):
    if label == 'J':
        return 0
    elif label == 'I':
        return 1
    elif label == 'H':
        return 2
    elif label == 'G':
        return 3
    elif label == 'F':
        return 4
    elif label == 'E':
        return 5
    elif label == 'D':
        return 6


def trans_clarity(label):
    if label == 'I1':
        return 0
    elif label == 'SI2':
        return 1
    elif label == 'SI1':
        return 2
    elif label == 'VS2':
        return 3
    elif label == 'VS1':
        return 4
    elif label == 'VVS2':
        return 5
    elif label == 'VVS1':
        return 6
    elif label == 'IF':
        return 7 



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------




def create_one_hot(df):
    """
    create_one_hot takes in diamonds and returns a DataFrame of one-hot 
    encoded features with names one_hot_<col>_<val> where <col> is the 
    original categorical column name, and <val> is the value found in 
    the categorical column <col>.

    :Example:
    >>> diamonds = pd.read_csv(os.path.join('data', 'diamonds.csv'))
    >>> out = create_one_hot(diamonds)
    >>> out.shape == (53940, 20)
    True
    >>> out.columns.str.startswith('one_hot').all()
    True
    >>> out.isin([0, 1]).all().all()
    True
    """
    new_df = pd.DataFrame()

    diamond_cols = ['cut', 'color','clarity']
    for k in diamond_cols:
        uniques = df[k].unique()
        for n in uniques:
            new_df['one_hot_' + k + '_' + n] = (df[k] == n).astype(int)
    return new_df



def create_proportions(df):
    """
    create_proportions takes in diamonds and returns a 
    DataFrame of proportion-encoded features with names 
    proportion_<col> where <col> is the original 
    categorical column name.

    >>> diamonds = pd.read_csv(os.path.join('data', 'diamonds.csv'))
    >>> out = create_proportions(diamonds)
    >>> out.shape[1] == 3
    True
    >>> out.columns.str.startswith('proportion_').all()
    True
    >>> ((out >= 0) & (out <= 1)).all().all()
    True
    """
    new_df = pd.DataFrame()

    diamond_cols = ['cut', 'color','clarity']
    for k in diamond_cols:
            props = df[k].value_counts() / df.shape[0]
            for n in props.index.values:
                df.replace(n, props.loc[n],inplace =True)
            new_df['proportion_' + k] = df[k]
    return new_df



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def create_quadratics(df):
    """
    create_quadratics that takes in diamonds and returns a DataFrame 
    of quadratic-encoded features <col1> * <col2> where <col1> and <col2> 
    are the original quantitative columns 
    (col1 and col2 should be distinct columns).

    :Example:
    >>> diamonds = pd.read_csv(os.path.join('data', 'diamonds.csv'))
    >>> out = create_quadratics(diamonds)
    >>> out.columns.str.contains(' * ').all()
    True
    >>> ('x * z' in out.columns) or ('z * x' in out.columns)
    True
    >>> out.shape[1] == 15
    True
    """
    diamond_quad_cols = ['carat','depth','table','x','y','z']
    permutes = list(itertools.combinations(diamond_quad_cols,2))
    new_df = pd.DataFrame()
    for n in permutes:
        new_df[n[0] + ' * ' + n[1]] = df[n[0]] * df[n[1]]
    return new_df


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------



def comparing_performance():
    """
    Hard-coded answers to comparing_performance.
    :Example:
    >>> out = comparing_performance()
    >>> len(out) == 6
    True
    >>> import numbers
    >>> isinstance(out[0], numbers.Real)
    True
    >>> all(isinstance(x, str) for x in out[2:-1])
    True
    >>> out[1] > out[-1]
    True
    """

    # create a model per variable => (variable, R^2, RMSE) table
    return [0.8493305264354858,1548.5331930613174, 'x','carat * x', 'ordinal_color',1434.8400089047332 ]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


class TransformDiamonds(object):
    
    def __init__(self, diamonds):
        self.data = diamonds
        
    # Question 6.1
    def transform_carat(self, data):
        """
        transform_carat takes in a DataFrame like diamonds 
        and returns a binarized carat column (an np.ndarray).
        :Example:
        >>> diamonds = pd.read_csv(os.path.join('data', 'diamonds.csv'))
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transform_carat(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> transformed[172, 0] == 1
        True
        >>> transformed[0, 0] == 0
        True
        """
        bi = Binarizer(threshold=1)

        binarized_carats = bi.transform(data[['carat']])
        return binarized_carats

    
    # Question 6.2
    def transform_to_quantile(self, data):
        """
        transform_to_quantiles takes in a DataFrame like diamonds 
        and returns an np.ndarray of quantiles of the weight 
        (i.e. carats) of each diamond.
        :Example:
        >>> diamonds = pd.read_csv(os.path.join('data', 'diamonds.csv'))
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transform_to_quantile(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> np.isclose(transformed[0, 0], 0.0075757, atol=0.0001)
        True
        >>> np.isclose(transformed[1, 0], 0.0025252, atol=0.0001)
        True
        """
        
        quant_trans = QuantileTransformer(n_quantiles=100)
        quant_trans = quant_trans.fit(self.data[['carat']])

        return quant_trans.transform(data[['carat']])
    
    # Question 6.3
    def transform_to_depth_pct(self, data):
        """
        transform_to_volume takes in a DataFrame like diamonds 
        and returns an np.ndarray consisting of the approximate 
        depth percentage of each diamond.
        :Example:
        >>> diamonds = pd.read_csv(os.path.join('data', 'diamonds.csv')).drop(columns='depth')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transform_to_depth_pct(diamonds)
        >>> len(transformed.shape) == 1
        True
        >>> np.isclose(transformed[0], 61.286, atol=0.0001)
        True
        """
        func_transform = FunctionTransformer(func=depth_func)
        X = data[['x','y','z']]
        X = np.transpose(X.to_numpy())
        return func_transform.transform(X)


def depth_func(X):
    #tpose = np.transpose(X)
    return ((2 * X[2][0:]) / (X[1][0:] + X[0][0:])) * 100
