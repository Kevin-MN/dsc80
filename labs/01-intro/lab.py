# lab.py


import os
import io
from tempfile import tempdir
import pandas as pd
import numpy as np
import doctest


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    adjacent elements that are consecutive integers.
    :Example:
    >>> consecutive_ints([5, 3, 6, 4, 9, 8])
    True
    >>> consecutive_ints([1, 3, 5, 7, 9])
    False
    >>> consecutive_ints([])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
    '''
    median_vs_mean takes in a non-empty list of numbers
    and returns a Boolean of whether the median is
    less than or equal to the mean.
    :param nums: a non-empty list of numbers.
    :returns: bool, whether the median is less than or equal to the mean.
    
    :Example:
    >>> median_vs_mean([6, 5, 4, 3, 2])
    True
    >>> median_vs_mean([50, 20, 15, 40])
    True
    >>> median_vs_mean([1, 8, 9])
    False
    '''
    
    temp_mean = mean(nums)
    temp_median = median(nums)
    if(temp_median <= temp_mean):
        return True
    else:
        return False


def mean(nums):
    return sum(nums) / len(nums)



def median(nums):
    sorted_nums = sorted(nums)
    if((len(nums) % 2) == 0):
        #print(sorted_nums[int(len(nums) / 2)],int(len(nums) / 2) )
        #print(sorted_nums[int((len(nums) / 2)) - 1], int((len(nums) / 2)) - 1)
        return (sorted_nums[int(len(nums) / 2)] + sorted_nums[int((len(nums) / 2) - 1)]) / 2
    else:
        return sorted_nums[int(len(nums) / 2)]




# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i positions apart, whose absolute 
    difference as integers is also i.
    :param ints: a list of integers
    :returns: a bool, describing whether or not the 
    above condition is satisfied
    :Example:
    >>> same_diff_ints([5, 3, 1, 5, 9, 8])
    True
    >>> same_diff_ints([1, 3, 5, 7, 9])
    False
    """
    

    if len(ints) == 0 or len(ints) == 1:
        return False 

    for k in range(len(ints) - 1):
        for n in range(k + 1,len(ints)):
            #print(k , n)
            diff = abs(ints[k] - ints[n])
            
            if diff == abs((k - n)):
                return True

    return False


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    """
    n_prefixes returns a string of n
    consecutive prefix of the input string.

    :param s: a string.
    :param n: an integer

    :returns: a string of n consecutive prefixes of s backwards.
    :Example:
    >>> n_prefixes('Billy', 4)
    'BillBilBiB'
    >>> n_prefixes('Marina', 3)
    'MarMaM'
    >>> n_prefixes('aaron', 2)
    'aaa'
    >>> n_prefixes('Justin', 5)
    'JustiJustJusJuJ'
    """
    
    concat_string = ""
    for i in range(n+1):
        concat_string = s[0:i] + concat_string 

    return concat_string


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    """
    exploded_numbers returns a list of strings of numbers from the
    input array each exploded by n.
    Each integer is zero padded.

    :param ints: a list of integers.
    :param n: a non-negative integer.

    :returns: a list of strings of exploded numbers. 
    :Example:
    >>> exploded_numbers([3, 8, 15], 2)
    ['01 02 03 04 05', '06 07 08 09 10', '13 14 15 16 17']
    >>> exploded_numbers([9, 99], 3)
    ['006 007 008 009 010 011 012', '096 097 098 099 100 101 102']
    """
    
    max_digits = len(str(ints[-1] + n))
    #print(max_digits)

    explode_list = []
    
    for i in range(len(ints)):
        temp_string = str(ints[i]).zfill(max_digits)
        for k in range(1,n+1):
            temp_string = str(ints[i] - k).zfill(max_digits) + ' ' + temp_string + ' ' + str(ints[i] + k).zfill(max_digits)
        explode_list.append(temp_string)

    return explode_list






# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of each line.
    :param fh: a file object to read from.
    :returns: a string of last characters from fh
    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """

    last_chars = ''
    for line in fh:
        last_chars += str(line)[-2]

    fh.close()
    return last_chars
    



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def add_root(A):
    """
    add_root takes in a numpy array and
    adds to each element the square-root of
    the index of each element.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = add_root(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    >>> np.isclose(out[3], 7 + np.sqrt(3))
    True
    """

    new_arr = np.array([])
    for i in range(len(A)):
        #print('iterate')
        new_arr = np.append(new_arr, float(A[i]) +  np.sqrt(i))

    return new_arr

def where_square(A):
    """
    where_square takes in a numpy array of integers
    and returns an array of Booleans
    whose ith element is True if and only if the ith element
    of the input array is a perfect square.
    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.
    :Example:
    >>> out = where_square(np.array([1, 2, 16, 17, 32, 49]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    >>> out[2]
    True
    """
    bool_arr = np.array([], dtype = bool)

    for i in range(len(A)):
        if int(str(np.sqrt(A[i])).split('.')[1]) != 0:
            bool_arr = np.append(bool_arr, False)
        else: 
            bool_arr = np.append(bool_arr, True)

    return bool_arr



# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    """
    growth_rates takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = growth_rates(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """
    
    grow_rates = np.array([])

    for i in range(1,len(A)):
        grow_rates = np.append(grow_rates, round((A[i] - A[i-1]) / A[i-1], 2))

    return grow_rates

    


def with_leftover(A):
    """
    Create a function with_leftover that takes in A and 
    returns the day on which you can buy at least 
    one share from 'left-over' money. If this never 
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: the day on which you can buy at least one share from 'left-over' money
    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = with_leftover(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """
    left_overs = 20 % A
    left_over_sum = np.cumsum(left_overs)

    for i in range(len(left_over_sum)):
        if(left_over_sum[i]) >= A[i]:
            return i

    return -1






# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def salary_stats(salary):
    """
    salary_stats returns a series as specified in the notebook.
    :param salary: a DataFrame of NBA salaries as found in `salary.csv`
    :return: a series with index specified in the notebook.
    :Example:
    >>> salary_fp = os.path.join('data', 'salary.csv')
    >>> salary = pd.read_csv(salary_fp)
    >>> out = salary_stats(salary)
    >>> isinstance(out, pd.Series)
    True
    >>> 'total_highest' in out.index
    True
    >>> isinstance(out.loc['duplicates'], bool)
    True
    """

    fifth_lowest_name = salary.sort_values(by= 'Salary').iloc[4][0]
    fifth_lowest_team = salary.sort_values(by= 'Salary').iloc[4][2]
    fifth_lowest = fifth_lowest_name + ", " + fifth_lowest_team

    temp_df = pd.DataFrame(salary['Player'].apply(extract_last)).groupby('Player').count().shape[0]

    orig_count = salary.shape[0]

    highest_paid_team = salary.loc[salary['Salary'] == np.max(salary['Salary']), 'Team'].iloc[0]
    total_highest = np.sum(salary.loc[salary['Team'] == highest_paid_team, 'Salary'])
    return pd.Series({'num_players': salary.shape[0], 'num_teams': salary.groupby("Team").count().shape[0], 'total_salary': np.sum(salary['Salary']), 
    'highest_salary': salary.loc[salary['Salary'] == np.max(salary['Salary']), 'Player'].iloc[0], 'avg_los': np.round(np.mean(salary.loc[salary['Team'] == 'Los Angeles Lakers', 'Salary']),2), 'fifth_lowest': fifth_lowest, 'duplicates': not(temp_df == orig_count), 'total_highest' : total_highest})

def extract_last(full_name):
    return full_name.split(' ')[1]
# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    """
    Parses and loads the malformed .csv file into a 
    properly formatted DataFrame (as described in 
    the question).
    :param fh: file handle for the malformed .csv file.
    :returns: a pandas DataFrame of the data, 
    as specified in the question statement.
    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """
   
    opened_file = open(fp)
    df = pd.DataFrame()
    labels = ['first', 'last', 'weight', 'height', 'geo']
    #print(labels)
    first = opened_file.readline()
    for line in opened_file:
        data = line.split(',')
        
        if len(data) > 6:
            #print(data)
            if '' in data:
                data.remove('')
            if '\n' in data:
                data.remove('\n')
            #print(data)

        for i in range(len(data)):
            data[i] = data[i].replace('\"', '')
        temp_row = { labels[0]:data[0], labels[1]:data[1], labels[2]:float(data[2]), labels[3]:float(data[3]), labels[4]:data[4] + "," + data[5].replace('\n', '') }
        df = df.append(temp_row, ignore_index = True)
    return df


