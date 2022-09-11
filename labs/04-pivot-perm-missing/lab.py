# lab.py


import pandas as pd
import numpy as np
import io
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def latest_login(login):
    """Calculates the latest login time for each user
    :param login: a DataFrame with login information
    :return: a DataFrame with latest login time for
    each user indexed by "Login Id"
    >>> fp = os.path.join('data', 'login_table.csv')
    >>> login = pd.read_csv(fp)
    >>> result = latest_login(login)
    >>> len(result)
    433
    >>> result.loc[381, "Time"].hour > 12
    True
    """
    
    login['Times'] = pd.to_datetime(login['Time'])
    login['Time']  = login['Times'].apply(extract_time)
    
    #login['Times'] = pd.to_datetime(login['Times'])
    
    #login['Time'] = login['Times']
    login = login.drop(columns = 'Times')
    #login['time3'] = login['Time']
    return login.groupby('Login Id').max()


def extract_time(date_time):
    return date_time.time()

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def smallest_elapsed(login):
    """
    Calculates the the smallest time elapsed for each user.
    :param login: a DataFrame with login information but without unique IDs
    :return: a DataFrame, indexed by Login ID, containing 
    the smallest time elapsed for each user.
    >>> fp = os.path.join('data', 'login_table.csv')
    >>> login = pd.read_csv(fp)
    >>> result = smallest_elapsed(login)
    >>> len(result)
    238
    >>> 18 < result.loc[1233, "Time"].days < 23
    True
    """
    more_1 = (login.groupby('Login Id').count() > 1)['Time'].values
   # more_1

    more_1_index = login.groupby('Login Id').count().loc[more_1].index
    #len(more_1_index)

    more_1_df = login.set_index('Login Id').loc[more_1_index].reset_index()#[more_1]
    more_1_df['Times'] = pd.to_datetime(more_1_df['Time'])
    times = pd.DataFrame(more_1_df.groupby('Login Id').Time.apply(np.array, dtype =np.datetime64))#.apply(find_min)
    times['Time'] = times['Time'].apply(np.diff)
    times['Time'] = times['Time'].apply(min)
    return times


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def total_seller(sales):
    """
    total_seller should take in the sales DataFrame and 
    return a DataFrame that contains the total sales 
    for each seller, indexed by 'Name'. There should not be any NaNs.

    >>> fp = os.path.join('data', 'sales.csv')
    >>> sales = pd.read_csv(fp)
    >>> out = total_seller(sales)
    >>> out.shape[0]
    3
    >>> out["Total"].sum() < 15000
    True
    """
    return sales.groupby('Name').sum()


def product_name(sales):
    """
    product_name should take in the sales DataFrame and 
    return a DataFrame that contains the total sales 
    for each seller, indexed by 'Product'. 
    Do not fill in NaNs.
    
    >>> fp = os.path.join('data', 'sales.csv')
    >>> sales = pd.read_csv(fp)
    >>> out = product_name(sales)
    >>> out.size
    15
    >>> out.loc["pen"].isnull().sum()
    0
    """
    return sales.pivot_table(index = 'Product', columns = 'Name', values = 'Total', aggfunc = 'sum' )


def count_product(sales):
    """
    count_product should take in the sales DataFrame and 
    return a DataFrame that contains the total number of 
    items sold product-wise and name-wise per date. 
    Replace NaNs with 0s.

    >>> fp = os.path.join('data', 'sales.csv')
    >>> sales = pd.read_csv(fp)
    >>> out = count_product(sales)
    >>> out.loc["boat"].loc["Trump"].value_counts()[0]
    6
    >>> out.size
    70
    """
    return sales.pivot_table(index = ['Product','Name'], columns = 'Date', values = 'Product', aggfunc = 'count' ).fillna(0)


def total_by_month(sales):
    """
    total_by_month should take in the sales DataFrame 
    and return a pivot table that contains the total 
    sales name-wise, product-wise per month. 
    Replace NaNs with 0s.
    
    >>> fp = os.path.join('data', 'sales.csv')
    >>> sales = pd.read_csv(fp)
    >>> out = total_by_month(sales)
    >>> out["May"].idxmax()
    ('Smith', 'book')
    >>> out.shape[1]
    5
    """
    sales['Month']= sales['Date'].apply(get_month)
    #sales
    out1 = sales.pivot_table(index = ['Name','Product'], columns = 'Month', values = 'Total', aggfunc = 'sum' ).fillna(0)
    return out1


def get_month(dt):
    month = dt[0:2]
    if month ==  '01':
        return 'January'
    elif month == '02':
        return 'Feburary'
    elif month == '03':
        return 'March'
    elif month == '04':
        return 'April'
    elif month == '05':
        return 'May'
    elif month == '06':
        return 'June'
    elif month == '07':
        return 'July'
    elif month == '08':
        return 'August'
    elif month == '09':
        return 'September'
    elif month == '10':
        return 'October'
    elif month == '11':
        return 'November'
    else:
        return 'December'
# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    """
    diff_of_means takes in a DataFrame like skittles 
    and returns the absolute difference of means 
    between the number of oranges per bag from Yorkville and Waco.
    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> out = diff_of_means(skittles)
    >>> 0 <= out
    True
    """
    fact_means = data.groupby('Factory').mean()[col].values
    return np.abs(fact_means[0] - fact_means[1])


def simulate_null(data, col='orange'):
    """
    simulate_null takes in a DataFrame like skittles and 
    returns one simulated instance of the test statistic 
    under the null hypothesis.
    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> out = simulate_null(skittles)
    >>> isinstance(out, float)
    True
    >>> 0 <= out <= 1.0
    True
    """
    shuffled_weights = (
        data['Factory']
        .sample(frac=1)
        .reset_index(drop=True)
    )

    shuffled = (
        data
        .assign(**{'Factory': shuffled_weights})
    )

    return diff_of_means(shuffled,col)


def pval_color(data, col='orange'):
    """
    pval_color takes in a DataFrame like skittles and 
    calculates the p-value for the permutation test 
    using 1000 trials.
    
    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> pval = pval_color(skittles)
    >>> isinstance(pval, float)
    True
    >>> 0 <= pval <= 0.1
    True
    """
    observed = diff_of_means(data, col)
    observed


    test_stats = np.array([])
    for n in np.arange(1000):
        test_stat = simulate_null(data,col)
        
        test_stats = np.append(test_stats, test_stat)

    return (test_stats >= observed).mean()


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def ordered_colors():
    """
    ordered_colors returns your answer as an ordered
    list from "most different" to "least different" 
    between the two locations. You list should be a 
    hard-coded list, where each element has the 
    form (color, p-value).

    :Example:
    >>> out = ordered_colors()
    >>> len(out) == 5
    True
    >>> colors = {'green', 'orange', 'purple', 'red', 'yellow'}
    >>> set([x[0] for x in out]) == colors
    True
    >>> all([isinstance(x[1], float) for x in out])
    True
    """
    return [('yellow', 0.000), ('orange', 0.043), ('red', 0.241), ('green', 0.494), ('purple', 0.985)]

# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------



def same_color_distribution():
    """
    same_color_distribution outputs a hard-coded tuple 
    with the p-value and whether you 'Fail to Reject' or 'Reject' 
    the null hypothesis.

    >>> out = same_color_distribution()
    >>> isinstance(out, tuple)
    >>> isinstance(out[0], float)
    True
    >>> out[1] in ['Fail to Reject', 'Reject']
    True
    """
    return [0.006, 'Reject']


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def perm_vs_hyp():
    """
    Multiple choice response for Question 8.

    >>> out = perm_vs_hyp()
    >>> ans = ['P', 'H']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    """
    return ['P','P', 'H','H','P']


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def after_purchase():
    """
    Multiple choice response for question 8

    >>> out = after_purchase()
    >>> ans = ['MD', 'MCAR', 'MAR', 'NI']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    """
    return ['NI', 'MD', 'MAR', 'NI','MAR']


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def multiple_choice():
    """
    Multiple choice response for question 9

    >>> out = multiple_choice()
    >>> ans = ['MD', 'MCAR', 'MAR', 'NI']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    >>> out[1] in ans
    True
    """
    return ['MAR','MAR','MD', 'NI', 'MCAR' ]
