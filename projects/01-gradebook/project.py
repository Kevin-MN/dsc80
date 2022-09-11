# project.py


import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:
    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint
    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    cols = np.array(grades.columns)
    str_arr = cols.astype(str)
   
    lab = []
    project = []
    midterm = []
    final = []
    disc = []
    checkpoint = [] 
    
    for i in range(len(str_arr)):
        #print('in loop')
        if(str_arr[i].startswith('lab') and len(str_arr[i]) == 5):
            lab.append(str(str_arr[i]))
           
        if(str_arr[i].startswith('project') and len(str_arr[i]) == 9):
             project.append(str(str_arr[i]))
            
        if(str_arr[i] == 'Midterm'):
             midterm.append(str(str_arr[i]))
           
        if(str_arr[i] == 'Final'):
            final.append(str(str_arr[i]))
           
        if(str_arr[i].startswith('discussion') and len(str_arr[i]) == 12):
            disc.append(str(str_arr[i]))
                
        if(str_arr[i].startswith('project') and str_arr[i].find('_checkpoint') != -1 and len(str_arr[i]) == 22):
            checkpoint.append(str(str_arr[i]))
         
        
        
    assignment_dict = {'lab': lab,'project':project,'midterm':midterm,'final':final,'disc':disc, 'checkpoint': checkpoint}
    return assignment_dict
         


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total takes in a DataFrame grades and returns the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    
    assign_cols = get_assignment_names(grades)

    project_free = np.array(grades.columns)

    all_project_cols = []
    for i in project_free:
        #print(i)
        if i.find('project') != -1:
            #print('appened')
            all_project_cols.append(i)
    


        
    #project_free = project_free[all_project_cols]
    #project_free
    #print(len(all_project_cols))

    filter_proj_cols = []
    #print(type(filter_proj_cols))

    for i in np.arange(len(all_project_cols)):
        if all_project_cols[i].find('Late') == -1 and all_project_cols[i].find('checkpoint') == -1:
            filter_proj_cols.append(all_project_cols[i])

    grades = grades[sorted(filter_proj_cols)]
    #grades

    #sorted_cols = grades_01 = grades.assign( project01 = grades['project01'].apply(convert_nan))

    sorted_cols = sorted(filter_proj_cols)
    grades_01 = grades[sorted(filter_proj_cols)]
    #grades.dtypes


    for i in filter_proj_cols:
        grades_01[i] = grades_01[i].apply(convert_nan)
    
    #i = 10
    #grades_01['new' + str(i) ] = grades_01['project01'] /grades_01['project01 - Max Points']
    #grades_01
    counter = 0
    #print(grades_01.dtypes)
    #print(len(sorted_cols))
    for i in np.arange(1,len(assign_cols['project']) + 1):
            #print('loop')
            #print(sorted_cols[counter][7:9])
            #print(sorted_cols[counter+2][7:9])
            #print(counter)
        if(sorted_cols[counter][7:9] == sorted_cols[counter+2][7:9]):
            grades_01['project' + str(i) + ' total'] = (grades_01[sorted_cols[counter]] + grades_01[sorted_cols[counter+2]]) / (grades_01[sorted_cols[counter+1]] + grades_01[sorted_cols[counter+3]])
            counter = counter + 4
            #print(counter)
            #print('has free response')
        else:
            grades_01['project' + str(i) + ' total'] = (grades_01[sorted_cols[counter]] / grades_01[sorted_cols[counter+1]])
            counter+=2
            #print(counter)
    #grades_01

    grades_01['Total'] = grades_01['project1 total']
    for i in np.arange(2,len(assign_cols['project']) + 1):
        grades_01['Total'] = grades_01['Total'] + grades_01['project' + str(i) + ' total']
    #grades_01

    return grades_01['Total'] / len(assign_cols['project'])
    #grades

def convert_nan(number):
    if(np.isnan(number)):
        return 0
    return number


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in a DataFrame 
    grades and returns a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by students that were marked "late" by Gradescope.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1, 10)])
    True
    >>> (out > 0).sum()
    8
    """
    
    assign_cols = get_assignment_names(grades)
    #assign_cols['lab']
    #assign_cols2 = 
    labs = assign_cols['lab']

    lab_names = assign_cols['lab']

    for i in np.arange(len(lab_names)):
        lab_names[i] = lab_names[i] + " - Lateness (H:M:S)"

    #print(lab_names)
    grades2 = grades[lab_names]
    #grades2

    #grades2

    for i in lab_names:
        grades2[i] = grades2[i].apply(convert_mins)


    counts = []    
    
    for i in np.arange(len(lab_names)):
        counts.append(grades2[lab_names[i]].sum()) 
    #counts

    for i in np.arange(len(lab_names)):
        lab_names[i] = lab_names[i].strip(' - Lateness (H:M:S)')

    ret_ser = pd.Series(index = labs, data = counts)



    #print(sorted(grades2['lab01 - Lateness (H:M:S)'].unique()))
    #print(sorted(grades2['lab02 - Lateness (H:M:S)'].unique()))
    #print(sorted(grades2['lab03 - Lateness (H:M:S)'].unique()))
    #print(sorted(grades2['lab04 - Lateness (H:M:S)'].unique()))
    #print(sorted(grades2['lab05 - Lateness (H:M:S)'].unique()))
    #print(sorted(grades2['lab06 - Lateness (H:M:S)'].unique()))
    #print(sorted(grades2['lab07 - Lateness (H:M:S)'].unique()))
    #print(sorted(grades2['lab08 - Lateness (H:M:S)'].unique()))
    #print(sorted(grades2['lab09 - Lateness (H:M:S)'].unique()))


    #counts
    return(ret_ser)





def convert_mins(time):
    nums = time.split(':')
    return (int(nums[0]) * 60 * 60 + int(nums[1]) * 60 + int(nums[2]) <= 20_000) and ((int(nums[0]) * 60 * 60) + (int(nums[1]) * 60) + int(nums[2]) != 0)




# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def lateness_penalty(col):
    """
    adjust_lateness takes in a Series containing
    how late a submission was processed
    and returns a Series of penalties according to the
    syllabus.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """

    
    #late
    late = col.apply(convert_mins2)
    #late
    late = late.apply(deter_penalty)
    return late


def convert_mins2(time):
    nums = time.split(':')
    return (int(nums[0]) * 60 * 60) + (int(nums[1]) * 60) + int(nums[2])

def deter_penalty(seconds):
    if seconds >= 0 and seconds <= 20_000:
        return 1.0
    elif seconds > 20_000 and seconds <= 604800:
        return 0.9
    elif seconds > 604800 and seconds <= (604800 * 2):
        return 0.7
    elif seconds > (604800 * 2):
        return 0.4




# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def process_labs(grades):
    """
    process_labs takes in a DataFrame like grades and returns
    a DataFrame of processed lab scores. The output should:
      * have the same index as `grades`,
      * have one column for each lab assignment (e.g. `'lab01'`, `'lab02'`,..., `'lab09'`),
      * have values representing the final score for each lab assignment, 
        adjusted for lateness and scaled to a score between 0 and 1.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1, 10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """

    assignments = get_assignment_names(grades)
    grades_clean_nan = grades[assignments['lab']] #.applymap(convert_nan)
    df_late_apply = pd.DataFrame(index = grades.index)
    for i in assignments['lab']:
        df_late_apply[i] = (grades_clean_nan[i]  / grades[i + ' - Max Points'])  * lateness_penalty(grades[i + ' - Lateness (H:M:S)'])



    return df_late_apply





# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def lab_total(processed):
    """
    lab_total takes in DataFrame of processed assignments (like the output of 
    Question 5) and returns a Series containing the total lab grade for each 
    student according to the syllabus.
    
    Your answers should be proportions between 0 and 1.
    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    
    processed = processed.fillna(0)
    return (processed.sum(axis = 1) - processed.min(axis = 1)) / (processed.shape[1] - 1) 
 


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def total_points(grades):
    """
    total_points takes in a DataFrame grades and returns a Series
    containing each student's course grade.
    Course grades should be proportions between 0 and 1.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    grades = grades.fillna(0)
    return ((assign_total(grades,'checkpoint') * 2.5) + (assign_total(grades,'disc') * 2.5) + ((grades['Midterm'] / grades['Midterm - Max Points']) * 15) +((grades['Final'] / grades['Final - Max Points']) * 30) + (projects_total(grades) * 30) + (lab_total(process_labs(grades) * 20))) / 100 


def assign_total(grades, assign_type):
    assign_cols = get_assignment_names(grades)

    project_free = np.array(grades.columns)

    all_project_cols = []
    for i in project_free:
        if i.find(assign_type) != -1:
            all_project_cols.append(i)
    


    filter_proj_cols = []

    for i in np.arange(len(all_project_cols)):
        if all_project_cols[i].find('Late') == -1:
            filter_proj_cols.append(all_project_cols[i])

    grades = grades[sorted(filter_proj_cols)]
    #grades

    #sorted_cols = grades_01 = grades.assign( project01 = grades['project01'].apply(convert_nan))

    sorted_cols = sorted(filter_proj_cols)
    grades_01 = grades[sorted(filter_proj_cols)]
    #grades.dtypes


    for i in filter_proj_cols:
        grades_01[i] = grades_01[i].apply(convert_nan)
    
    counter = 0
    for i in np.arange(1,len(assign_cols[assign_type]) + 1):
            grades_01[assign_type + str(i) + ' total'] = (grades_01[sorted_cols[counter]] / grades_01[sorted_cols[counter+1]])
            counter+=2
       

    grades_01['Total'] = grades_01[assign_type +'1 total']
    for i in np.arange(2,len(assign_cols[assign_type]) + 1):
        grades_01['Total'] = grades_01['Total'] + grades_01[assign_type + str(i) + ' total']
    #grades_01

    return grades_01['Total'] / len(assign_cols[assign_type])
    #grades

# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.
    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    temp_df = pd.DataFrame(data = total, columns = ['0'])
    temp_df['0'] = temp_df['0'].apply(deter_letter)
    return temp_df['0']


def deter_letter(total):
    if(total >= 0.9):
        return 'A'
    elif (total >= 0.8) & (total < 0.9):
        return 'B'
    elif (total >= 0.7) & (total < 0.8):
        return 'C'
    elif (total >= 0.6) & (total < 0.7):
        return 'D'
    elif (total < 0.6):
        return 'F'

def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    out_put = final_grades(total_points(grades))
    return out_put.value_counts().sort_values(ascending = False) / out_put.shape[0]


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def simulate_pval(grades, N):
    """
    simulate_pval takes in a DataFrame grades and
    a number of simulations N and returns the p-value
    for the hypothesis test described in the notebook.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 1000)
    >>> 0 <= out <= 0.1
    True
    """
    
    df_gradecheck = pd.DataFrame()
    df_gradecheck['grades'] = total_points(grades)
    df_gradecheck['level'] = grades['Level']
    df_gradecheck = df_gradecheck.groupby('level').mean()



    observed_senior = df_gradecheck.loc['SR'][0]
    #observed_senior

    num_seniors = grades['Level'].value_counts().loc['SR']
    #num_seniors

    test_stats = np.random.choice(total_points(grades), size = (N,num_seniors), replace = True)
    test_stats = test_stats.mean(axis = 1) 
    test_stats = test_stats <= observed_senior
    return test_stats.mean()



# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    grades = grades.fillna(0)
    return ((assign_total_normal(grades,'checkpoint') * 2.5) + (assign_total_normal(grades,'disc') * 2.5) + (np.clip(((grades['Midterm'] / grades['Midterm - Max Points']) + np.random.normal(0, 0.02, size = (grades.shape[0]))),0,1) * 15) +(np.clip(((grades['Final'] / grades['Final - Max Points']) + np.random.normal(0, 0.02, size = (grades.shape[0]))),0,1) * 30) + (projects_total_normal(grades) * 30) + (lab_total(process_labs_normal(grades) * 20))) / 100





def assign_total_normal(grades, assign_type):
    assign_cols = get_assignment_names(grades)

    project_free = np.array(grades.columns)

    all_project_cols = []
    for i in project_free:
        if i.find(assign_type) != -1:
            all_project_cols.append(i)
    


    filter_proj_cols = []

    for i in np.arange(len(all_project_cols)):
        if all_project_cols[i].find('Late') == -1:
            filter_proj_cols.append(all_project_cols[i])

    grades = grades[sorted(filter_proj_cols)]
    #grades

    #sorted_cols = grades_01 = grades.assign( project01 = grades['project01'].apply(convert_nan))

    sorted_cols = sorted(filter_proj_cols)
    grades_01 = grades[sorted(filter_proj_cols)]
    #grades.dtypes


    for i in filter_proj_cols:
        grades_01[i] = grades_01[i].apply(convert_nan)
    
    counter = 0
    for i in np.arange(1,len(assign_cols[assign_type]) + 1):
            grades_01[assign_type + str(i) + ' total'] = np.clip(((grades_01[sorted_cols[counter]] / grades_01[sorted_cols[counter+1]])) + np.random.normal(0, 0.02, size = (grades.shape[0])),0,1)
            counter+=2
       

    grades_01['Total'] = grades_01[assign_type +'1 total']
    for i in np.arange(2,len(assign_cols[assign_type]) + 1):
        grades_01['Total'] = grades_01['Total'] + grades_01[assign_type + str(i) + ' total']
    #grades_01

    return grades_01['Total'] / len(assign_cols[assign_type])
    #grades




def projects_total_normal(grades):
    '''
    projects_total takes in a DataFrame grades and returns the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    
    assign_cols = get_assignment_names(grades)

    project_free = np.array(grades.columns)

    all_project_cols = []
    for i in project_free:
        #print(i)
        if i.find('project') != -1:
            #print('appened')
            all_project_cols.append(i)
    


        
    #project_free = project_free[all_project_cols]
    #project_free
    #print(len(all_project_cols))

    filter_proj_cols = []
    #print(type(filter_proj_cols))

    for i in np.arange(len(all_project_cols)):
        if all_project_cols[i].find('Late') == -1 and all_project_cols[i].find('checkpoint') == -1:
            filter_proj_cols.append(all_project_cols[i])

    grades = grades[sorted(filter_proj_cols)]
    #grades

    #sorted_cols = grades_01 = grades.assign( project01 = grades['project01'].apply(convert_nan))

    sorted_cols = sorted(filter_proj_cols)
    grades_01 = grades[sorted(filter_proj_cols)]
    #grades.dtypes


    for i in filter_proj_cols:
        grades_01[i] = grades_01[i].apply(convert_nan)
    
    #i = 10
    #grades_01['new' + str(i) ] = grades_01['project01'] /grades_01['project01 - Max Points']
    #grades_01
    counter = 0
    #print(grades_01.dtypes)
    #print(len(sorted_cols))
    for i in np.arange(1,len(assign_cols['project']) + 1):
            #print('loop')
            #print(sorted_cols[counter][7:9])
            #print(sorted_cols[counter+2][7:9])
            #print(counter)
        if(sorted_cols[counter][7:9] == sorted_cols[counter+2][7:9]):
            grades_01['project' + str(i) + ' total'] = np.clip(((grades_01[sorted_cols[counter]] + grades_01[sorted_cols[counter+2]]) / (grades_01[sorted_cols[counter+1]] + grades_01[sorted_cols[counter+3]])) + + np.random.normal(0, 0.02, size = grades.shape[0]), 0,1)
            counter = counter + 4
            #print(counter)
            #print('has free response')
        else:
            grades_01['project' + str(i) + ' total'] = np.clip((grades_01[sorted_cols[counter]] / grades_01[sorted_cols[counter+1]]) + np.random.normal(0, 0.02, size = grades.shape[0]), 0,1)
            counter+=2
            #print(counter)
    #grades_01

    grades_01['Total'] = grades_01['project1 total']
    for i in np.arange(2,len(assign_cols['project']) + 1):
        grades_01['Total'] = grades_01['Total'] + grades_01['project' + str(i) + ' total']
    #grades_01

    return grades_01['Total'] / len(assign_cols['project'])
    #grades




def process_labs_normal(grades):
    """
    process_labs takes in a DataFrame like grades and returns
    a DataFrame of processed lab scores. The output should:
      * have the same index as `grades`,
      * have one column for each lab assignment (e.g. `'lab01'`, `'lab02'`,..., `'lab09'`),
      * have values representing the final score for each lab assignment, 
        adjusted for lateness and scaled to a score between 0 and 1.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1, 10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """

    assignments = get_assignment_names(grades)
    grades_clean_nan = grades[assignments['lab']] #.applymap(convert_nan)
    df_late_apply = pd.DataFrame(index = grades.index)
    for i in assignments['lab']:
        df_late_apply[i] = np.clip(((grades_clean_nan[i]  / grades[i + ' - Max Points'])  * lateness_penalty(grades[i + ' - Lateness (H:M:S)'])) + np.random.normal(0, 0.02, size = grades.shape[0]),0,1)



    return df_late_apply

# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.
    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 0.5 < out[2][0] < 1
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """
    out = [0.0010776552430662863, 0.8242990654205608, (0.7981308411214953, 0.8616822429906542), 0.0691588785046729, (False, False)]
    return out
