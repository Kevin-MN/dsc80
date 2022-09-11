# lab.py


import os
import io
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.
    
    :Example:
    >>> set(car_null_hypoth()) <= set(range(1, 7))
    True
    """
    return [1,3,4,5]


def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.
    
    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1, 7))
    True
    """
    return [2,6]

def car_test_stat():
    """
    Returns a list of valid test statistics.
    
    :Example:
    >>> set(car_test_stat()) <= set(range(1, 5))
    True
    """
    return [1,2,4]

def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.
    
    :Example:
    >>> car_p_value() in set(range(1, 6))
    True
    """
    return 5


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def clean_universities(df):
    """ 
    clean_universities takes in the raw rankings DataFrame
    and returns a cleaned DataFrame according to the instructions
    in the lab notebook.
    >>> fp = os.path.join('data', 'universities_unified.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_universities(df)
    >>> cleaned.shape[0] == df.shape[0]
    True
    >>> cleaned['nation'].nunique() == 59
    True
    """
    df2 = df.copy() #pd.read_csv('/home/v/Documents/github_repos/dsc80-2022-sp/labs/03-hyp-dataframes/data/universities_unified.csv')
    #df[:50]
    df2['institution'] = df2['institution'].str.replace('\n',', ')
    df2['broad_impact'] = df2['broad_impact'].astype(int)
    #df[:50]#.dtypes


    splits = df2['national_rank'].str.split(',')
    #splits
    df2[['nation', 'national_rank_cleaned']] = pd.DataFrame(splits.tolist())
    df2['national_rank_cleaned'] = df2['national_rank_cleaned'].astype(int)
    df2['nation'] = df2['nation'].apply(clean_3)
    df2 = df2.drop(columns = ['national_rank'])
    #df.dtypes

    bool_df = pd.DataFrame(index = df.index)

    bool_df['control_bool'] = df2['control'].apply(convert_0_1)
    bool_df['city_bool'] = df2['city'].apply(convert_0_1)
    bool_df['state_bool'] = df2['state'].apply(convert_0_1)
    #bool_df[1:50]              
        #,df['city'].apply(convert_0_1),df['state'].apply(convert_0_1) ])
    bool_df['public'] = df2['control'].apply(public)
    bool_df = bool_df.sum(axis = 1)
    bool_df = (bool_df == 4)
    df2['is_r1_public'] = bool_df
    return df2

def public(word):
    if (~pd.isnull(word)) & (word == 'Public'):
        return 1
    else:
        return 0


def convert_0_1(word):
    if pd.isnull(word):
        return 0
    else:
        return 1

def clean_3(word):
    if(word == 'USA'):
        return 'United States'
    elif word == 'UK':
        return 'United Kingdom'
    elif word == 'Czechia':
        return 'Czech Republic'
    else:
        return word


def university_info(cleaned):
    """
    university_info takes in a cleaned rankings DataFrame
    and returns a list containing the four values described
    in the lab notebook.
    >>> fp = os.path.join('data', 'universities_unified.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_universities(df)
    >>> info = university_info(cleaned)
    >>> len(info) == 4
    True
    >>> all([isinstance(x, y) for x, y in zip(info, [str, float, str, str])])
    True
    >>> info[2] in cleaned['state'].unique()
    True
    """
    one = cleaned[cleaned['citations'] <= 500].groupby('nation').mean().sort_values(by = 'world_rank').index[0]
    two = cleaned.loc[cleaned.sort_values(by = 'quality_of_education').index[cleaned.shape[0] - 200:cleaned.shape[0]], 'publications'].mean()
    three =  cleaned[cleaned['nation'] == 'United States'].groupby('state').mean().sort_values(by = 'is_r1_public', ascending = False).index[0]
    four = cleaned[cleaned['national_rank_cleaned'] == 1].sort_values(by = 'world_rank', ascending = False)['institution'].iloc[0]

    return [one, two, three, four]



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def std_scores_by_nation(cleaned):
    """
    std_scores_by_nation takes in a cleaned DataFrame of university rankings
    and returns a DataFrame containing standardized scores, according to
    the instructions in the lab notebook.
    >>> fp = os.path.join('data', 'universities_unified.csv')
    >>> play = pd.read_csv(fp)
    >>> cleaned = clean_universities(play)
    >>> out = std_scores_by_nation(cleaned)
    >>> out.shape[0] == cleaned.shape[0]
    True
    >>> set(out.columns) == set(['institution', 'nation', 'score'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """
    #df_cleaned = clean_universities(cleane)
    df_cleaned2 = pd.DataFrame(data = cleaned['institution'] , index = cleaned.index)
    df_cleaned2['nation'] = cleaned['nation']
    df_cleaned2['score'] = cleaned['score']

    df_cleaned2 = df_cleaned2.groupby('nation')
    df2 = df_cleaned2.transform(lambda x: (x - x.mean()) / x.std())


    #df2['score']
    #df_cleaned2.set_index('nation')

    #std = df_cleaned[df_cleaned['nation'] == 'China']['score'].std()
    #mean = df_cleaned[df_cleaned['nation'] == 'China']['score'].mean()

    #df
    #(44.02 - mean) / std

    df_cleaned3 = pd.DataFrame(data = cleaned['institution'] , index = cleaned.index)
    df_cleaned3['nation'] = cleaned['nation']
    df_cleaned3['score'] = df2['score']
    return df_cleaned3
        

def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0] in np.arange(1, 4)
    True
    >>> isinstance(out[1], str)
    True
    """
    return [2,'United States']


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname):
    """
    read_linkedin_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_linkedin_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_linkedin_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    
    file_list = os.listdir(dirname)

    list_of_dfs = [pd.read_csv(os.path.join(dirname, file)) for file in file_list]
    #list_of_dfs[1]

    df1 = list_of_dfs[0]
    #df1

    for i in list_of_dfs:
        old_cols = i.columns
        cols = []
        for j in old_cols: 
            cols.append(j.lower().replace('_', ' '))
            
        #print(j)
    #print(cols)
        i.columns = cols
    #list_of_dfs[2]
    #list_of_dfs[4]
    big_list = pd.concat(list_of_dfs)
    big_list = big_list.reset_index().drop(columns = 'index')

    sorted_cols_df = pd.DataFrame(data = big_list['first name'], index = big_list.index)
    sorted_cols_df['last name'] = big_list['last name']
    sorted_cols_df['current company'] = big_list['current company']
    sorted_cols_df['job title'] = big_list['job title']
    sorted_cols_df['email'] = big_list['email']
    sorted_cols_df['university'] = big_list['university']
    return sorted_cols_df



def com_stats(df):
    """
    com_stats 
    :param df: a DataFrame containing the combined survey data
    :returns: a hardcoded list of answers to the problems in the notebook
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_linkedin_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> isinstance(out[0], int)
    True
    >>> isinstance(out[2], str)
    True
    """
    one = df['current company'].value_counts().sort_values(ascending = False)[0]
    two = (df['email'].str.endswith('.edu') == True).sum()
    idx = df['job title'].astype(str).apply(len).sort_values(ascending = False).index[0]
    three = df.loc[idx, 'job title']

    dupe = df['job title']
    dupe = dupe.astype(str).apply(str.lower)
    four  = dupe[dupe.str.contains('manager')].shape[0]
    

    return [int(one),two, three, four]

def get_length(word):
    return len(str(word))


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    """
    read_student_surveys takes in a directory path 
    (containing files favorite*.csv) and combines 
    all of the survey data into one DataFrame, 
    indexed by student ID (a value 1-1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = read_student_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> read_student_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    file_list = os.listdir(dirname)
    #file_list

    list_of_dfs = [pd.read_csv(os.path.join(dirname, file)) for file in file_list]
    #list_of_dfs[0]

    big_df = list_of_dfs[0]
    for i in np.arange(1,len(list_of_dfs)):
        big_df = big_df.merge(list_of_dfs[i], on = 'id' )
    big_df = big_df.set_index('id')
    return big_df


def check_credit(df):
    """
    check_credit takes in a DataFrame with the 
    combined survey data and outputs a DataFrame 
    of the names of students and how many extra credit 
    points they would receive, indexed by their ID (a value 1-1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = read_student_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    >>> out['ec'].max()
    6
    """
    big_df = df.copy()
    names = big_df['name']
    #names
    #big_df
    big_df = big_df.drop(columns = ['name']).applymap(apply_nan)

    class_extra = big_df.sum(axis = 0)
    class_extra = class_extra / big_df.shape[0]
    class_extra = class_extra[class_extra >= 0.90].shape[0]

    if(class_extra >= 1):
        class_extra = 1
    else:
        class_extra = 0

    indiv = big_df.sum(axis = 1)
    indiv = indiv / len(big_df.columns)

    totals = ((indiv >= .75) * 5) + class_extra
    #otals
    #(indiv >= .75)

    final = pd.DataFrame(data = names, index = big_df.index)
    final['ec'] = totals
    return final




def apply_nan(word):
    if pd.isnull(word):
        return 0
    else:
        return 1

# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    """
    most popular 'ProcedureType'
    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = most_popular_procedure(pets, procedure_history)
    >>> isinstance(out, str)
    True
    """
    
    pets_and_hist = pets.merge(procedure_history, on = 'PetID', how = 'left')
    return pets_and_hist['ProcedureType'].value_counts().sort_values(ascending = False).index[0]


def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """
    
    names = owners.merge(pets, on = 'OwnerID', how = 'left', suffixes = ('_owner','_pet'))
    owner_names = pd.DataFrame([owners['Name'], owners['OwnerID']]).T
    names = names.groupby('OwnerID')['Name_pet'].apply(lambda x : ",".join(x)).reset_index()
    #names = names.merge(owner_names, on = )
    names['Name_pet'] = names['Name_pet'].apply(split_names)
    owner_names
    names = names.merge(owner_names, on = 'OwnerID', how = 'left').drop(columns = 'OwnerID').set_index('Name')
    #names['Name_pet']
    return names['Name_pet']
    #pd.DataFrame(names)



def split_names(word):
    if word.find(',') != -1:
        return word.split(';')
    else:
        return word


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    """
    total cost per city
​
    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_city(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['City'])
    True
    """
    history_wdetails = procedure_history.merge(procedure_detail, on = ['ProcedureType','ProcedureSubCode'])
    #history_wdetails

    pets_wdetails = pets.merge(history_wdetails, on = 'PetID', how = 'inner')
    #pets_wdetails


    with_cities = pets_wdetails.merge(owners, on = 'OwnerID')
    return with_cities.groupby('City').sum()['Price']
