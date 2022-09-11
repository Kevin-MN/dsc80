# lab.py


import pandas as pd
import numpy as np
import os
import doctest


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def data_load(scores_fp):
    """
    Follows outlined steps to create a DataFrame
    :param scores_fp: file name as a string
    :return: a DataFrame
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> isinstance(scores, pd.DataFrame)
    True
    >>> list(scores.columns)
    ['attempts', 'highest_score']
    >>> isinstance(scores.index[0], int)
    False
    """
    df = pd.read_csv(scores_fp, usecols = ['name', 'tries', 'highest_score', 'sex'])
    df = df.drop('sex',axis=1)
    df = df.rename(columns = {'name':'firstname', 'tries':'attempts'})
    df = df.set_index('firstname')
    return df


def pass_fail(scores):
    """
    Modifies the scores DataFrame by adding one more column satisfying
    conditions from the write up.
    :param scores: DataFrame from the question above
    :return: DataFrame with additional column pass
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> isinstance(scores, pd.DataFrame)
    True
    >>> len(scores.columns)
    3
    >>> scores.loc["Julia", "pass"] == 'Yes'
    True
    """
    

    scores['pass'] = ((scores['highest_score'] >= 50) & (scores['attempts'] < 3)) | ((scores['highest_score'] >= 70) & (scores['attempts'] < 6)) | ((scores['highest_score'] >= 90) & (scores['attempts'] < 10))
    scores['pass'] = scores['pass'].apply(apply_yes_no)
    return scores


def apply_yes_no(value):
    if value:
        return 'Yes'
    else:
        return 'No'


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def av_score(scores):
    """
    returns the average score for those students who passed the test.
    :param scores: DataFrame from the second question
    :return: average score
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> av = av_score(scores)
    >>> isinstance(av, float)
    True
    >>> 91 < av < 92
    True
    """
    
    passed = scores[scores['pass'] == 'Yes']
    return passed['highest_score'].mean()





def highest_score_name(scores):
    """
    finds the highest score and people who received it
    :param scores: DataFrame from the second question
    :return: dictionary where the key is the highest score and the value(s) is a list of name(s)
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> highest = highest_score_name(scores)
    >>> isinstance(highest, dict)
    True
    >>> len(list(highest.values())[0])
    3
    """
    
    df_sort = scores.sort_values(by = 'highest_score', ascending = False)
    high_score = df_sort['highest_score'][0]
    names = df_sort[df_sort['highest_score'] == high_score].index
    return {high_score:list(names)}


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def idx_dup():
    """
    Answers the question in the write up.
    :return:
    >>> ans = idx_dup()
    >>> isinstance(ans, int)
    True
    >>> 1 <= ans <= 6
    True
    """
    return 6


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def trick_me():
    """
    Answers the question in the write-up
    :return: a number
    >>> ans =  trick_me()
    >>> ans == 1 or ans == 2 or ans == 3
    True
    """
    return 3


def reason_dup():
    """
     Answers the question in the write-up
    :return: a number
    >>> ans =  reason_dup()
    >>> ans == 1 or ans == 2 or ans == 3
    True
    """
    return 2


def trick_bool():
    """
     Answers the question in the write-up
    :return: a list with three numbers
    >>> ans =  trick_bool()
    >>> isinstance(ans, list)
    True
    >>> isinstance(ans[1], int)
    True
    """
    
    return [4,10,13]

def reason_bool():
    """
    Answers the question in the write-up
    :return: a number
    >>> ans =  reason_bool()
    >>> ans == 1 or ans == 2 or ans == 3 or ans == 4
    True
    """
    
    return 2



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def change(x):
    """
    Returns 'MISSING' when x is `NaN`,
    Otherwise returns x
    >>> change(1.0) == 1.0
    True
    >>> change(np.NaN) == 'MISSING'
    True
    """
    if np.isnan(x):
        return "MISSING"
    else:
        return x 
        



def correct_replacement(nans):
    """
    changes all np.NaNs to "Missing"
    :param nans: given dataframe
    :return: modified dataframe
    >>> nans = pd.DataFrame([[0, 1, np.NaN], [np.NaN, np.NaN, np.NaN], [1, 2, 3]])
    >>> A = correct_replacement(nans)
    >>> (A.values == 'MISSING').sum() == 4
    True
    """
    
    nan_mod = nans.applymap(change)
    return nan_mod



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def population_stats(df):
    """
    population_stats which takes in a DataFrame df 
    and returns a DataFrame indexed by the columns 
    of df, with the following columns:
       * `'num_nonnull'` contains the number of non-null entries 
                         in each column.
       * `'prop_nonnull'` contains the proportion of entries in 
                          each column that are non-null.
       * `'num_distinct'` contains the number of distinct non-null 
                          entries in each column.
       * `'prop_distinct'` contains the proportion of non-null 
                           entries that are distinct in each column.
    :Example:
    >>> data = np.random.choice(range(10), size=(100, 4))
    >>> df = pd.DataFrame(data, columns='A B C D'.split())
    >>> out = population_stats(df)
    >>> out.index.tolist() == ['A', 'B', 'C', 'D']
    True
    >>> cols = ['num_nonnull', 'prop_nonnull', 'num_distinct', 'prop_distinct']
    >>> out.columns.tolist() == cols
    True
    >>> (out['num_distinct'] <= 10).all()
    True
    >>> (out['prop_nonnull'] == 1.0).all()
    True
    """
    
   
    new_df2 = df.applymap(change2)
    non_nulls = new_df2.T.sum(axis = 1)
    distincts = df.nunique()

    #print(non_nulls)
    #print(distincts)


    df_ret = pd.DataFrame(data = {'num_nonnull':non_nulls , 'prop_nonnull': non_nulls/df.shape[0], 'num_distinct':distincts, 'prop_distinct':distincts/non_nulls})
    return df_ret




def change2(x):
    if ~pd.isnull(x):
        return 1
    else:
        return 0
# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def most_common(df, N=10):
    """
    `most_common` which takes in a DataFrame df and returns 
    a DataFrame of the N most-common values and their counts
    for each column of df.
    :param df: input dataframe.
    :param N: number of most common elements to return (default 10).
    :Example:
    >>> data = np.random.choice(range(10), size=(100, 2))
    >>> df = pd.DataFrame(data, columns='A B'.split())
    >>> out = most_common(df, N=3)
    >>> out.index.tolist() == [0, 1, 2]
    True
    >>> out.columns.tolist() == ['A_values', 'A_counts', 'B_values', 'B_counts']
    True
    >>> out['A_values'].isin(range(10)).all()
    True
    """
    

    new_df = pd.DataFrame(index = np.arange(0,N))
    for i in list(df.columns):
        top_n = df[i].value_counts().sort_values(ascending = False)
    #print(top_n.index)
    
        if(top_n.shape[0] < N):
            temp = top_n.append(pd.Series(np.repeat(np.nan,N - top_n.shape[0]),index = np.repeat(np.nan,N - top_n.shape[0])))
            #print(type(temp))
    
            #print('temp',temp)
            new_df[i + '_values'] = temp.index[0:N]
            #print('after issue1')
        
            new_df[i + '_counts'] = temp.values[0:N]
            #print('after issue2')
        else:
            new_df[i + '_values'] = top_n.index[0:N]
            new_df[i + '_counts'] = top_n.values[0:N]


    return new_df#.dtypes


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def null_hyp():
    """
    :Example:
    >>> isinstance(null_hypoth(), list)
    True
    >>> set(null_hyp()).issubset({1, 2, 3, 4})
    True
    """
    return [1,2,4]


def simulate_null():
    """
    :Example:
    >>> pd.Series(simulate_null()).isin([0, 1]).all()
    True
    """
    return np.random.choice([0,1], 250,p=[.02,.98])


def estimate_p_val(N):
    """
    >>> 0 < estimate_p_val(1000) < 0.1
    True
    """

    result  = np.random.choice([0,1],size = (N,250),p=[.02,.98]).sum(axis = 1) / 250
    result = 1 - result
    #result
    return (result >= .04).mean()

# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    """
    `super_hero_powers` takes in a DataFrame like 
    powers and returns a list with the following three entries:
        - The name of the superhero with the greatest number of superpowers.
        - The name of the most common superpower among superheroes whose names begin with `'M'`.
        - The name of the most common superpower among superheroes with only one superpower.

    :Example:
    >>> fp = os.path.join('data', 'superheroes_powers.csv')
    >>> powers = pd.read_csv(fp)
    >>> out = super_hero_powers(powers)
    >>> isinstance(out, list)
    True
    >>> len(out)
    3
    >>> all([isinstance(x, str) for x in out])
    True
    """
    
    powers = powers.set_index('hero_names')
    greatest_num = powers.sum(axis=1).idxmax()
    #greatest_num

    M_common = powers.reset_index()
    M_common = M_common[M_common['hero_names'].str.startswith('M')].set_index('hero_names')
    M_common = M_common.sum(axis = 0).idxmax()
    #M_common

    one_super = powers
    one_super = one_super.sum(axis = 1)
    one_super = powers[one_super == 1]
    #one_super

    one_super = one_super.sum(axis = 0).idxmax()
    #one_super
    return [greatest_num,M_common,one_super]


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    """
    clean_heroes takes in the DataFrame heroes
    and replaces values that are 'null-value'
    place-holders with np.NaN.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = clean_heroes(heroes)
    >>> out['Skin color'].isnull().any()
    True
    >>> out['Weight'].isnull().any()
    True
    """

    heroes = heroes.applymap(clean_entries)
    return heroes


def clean_entries(enter):
    if  enter  == '-' or enter == -99.0:
        return np.nan
    else: 
        return enter


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def super_hero_stats():
    """
    Returns a list that answers the questions in the notebook.
    :Example:
    >>> out = super_hero_stats()
    >>> out[0] in ['Marvel Comics', 'DC Comics']
    True
    >>> isinstance(out[1], int)
    True
    >>> isinstance(out[2], str)
    True
    >>> out[3] in ['good', 'bad']
    True
    >>> isinstance(out[4], str)
    True
    >>> 0 <= out[5] <= 1
    True
    """
    return ['Marvel Comics', 99, 'Groot', 'bad', 'Onslaught', 0.28608247]


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def bhbe_col(heroes):
    """
    `bhbe` ('blond-hair-blue-eyes') returns a Boolean Series that contains `True` 
    for characters that have **both** blond hair and blue eyes, 
    and `False` for all other characters.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = bhbe_col(heroes)
    >>> isinstance(out, pd.Series)
    True
    >>> out.dtype == np.dtype('bool')
    True
    >>> out.sum()
    93
    """

    heroes['Hair color'] = heroes['Hair color'].apply(str.lower)
    heroes['Eye color'] = heroes['Eye color'].apply(str.lower)
     
    return (heroes['Hair color'].str.contains('blond')) & (heroes['Eye color'].str.contains('blue'))


# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def observed_stat(heroes):
    """
    observed_stat returns the observed test statistic
    for the hypothesis test.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = observed_stat(heroes)
    >>> 0.5 <= out <= 1.0
    True
    """

    all_bhbe = heroes[bhbe_col(heroes)]
    observed = all_bhbe['Alignment'].value_counts()
    return observed[0] / all_bhbe.shape[0]

    
    

def simulate_bhbe_null(n):
    """
    `simulate_null` takes in a number `n` and returns an array of length `n`, 
    where each element is a simulated test statistic according to the null hypothesis. 

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = simulate_bhbe_null(10)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.shape[0]
    10
    >>> ((0.45 <= out) & (out <= 1)).all()
    True
    """
    
    sims = np.random.choice([0,1,2], size = (n,734), p=(.6790,.2852,.0358))
    results = (sims[0:] == 0).sum(axis = 1) / 734
    return results

    

def calc_pval():
    """
    calc_pval returns a list where:
        - the first element is the p-value for 
        the hypothesis test (using 100,000 simulations).
        - the second element is 'Reject' if you reject 
        the null hypothesis and 'Fail to reject' if you 
        fail to reject the null hypothesis at the
        1% significance level.

    :Example:
    >>> out = calc_pval()
    >>> len(out)
    2
    >>> 0 <= out[0] <= 1
    True
    >>> out[1] in ['Reject', 'Fail to reject']
    True
    """
    return [0.0, 'Reject']
