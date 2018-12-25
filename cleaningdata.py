# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:36:47 2018

@author: Vivek
"""


''' CLEANING DATA '''


#%%

''' Importing dataset "dob_job_application_filings_subset.csv"'''

import pandas as pd

data = pd.read_csv('dob_job_application_filings_subset.csv')

#Print head of data
print(data.head())
#%%
#print tail of data

print(data.tail())
#%%
#print shape of data
print(data.shape)   # shape is an attribute not method hence no need of ()
#%%
#print columns of data
print(data.columns) # columns is an attribute not method hence no need of ()

#%%
#printing column name using loop
column = []
for word in data.columns:
    column.append(word)

print(column)
#%%
for word in data.columns:
    print(word,':',type(word))  #printing column name along with their data-Type
    
#%%

#print info of data
print(data.info())

#%%
#print to describe statistical parameters of dataset
#NOTE: .describe() method works only on numerica keysc

print(data.describe())

#%%
'''frequency counts of rows in dataset'''

data['City '].value_counts(dropna =0).head()  # gives counts of unique rows throughout the column

#%%
''' Visualizing data to find outliers'''

import matplotlib.pyplot as plt

exist_zone = data['Existing Zoning Sqft'].plot(kind='hist', rot=70,logx=True,logy=True)
plt.show()

#%%
import seaborn as sns
exist_zone1 = sns.distplot(data['Existing Zoning Sqft'],rug=True)
plt.show()
#%%
''' Using pandas to create box plot'''
data.boxplot(column='Initial Cost', by='Borough', rot=90)
plt.show()

#%%
'''Boxplots are great when you have a numeric column that you want to compare across different categories.
 When you want to visualize two numeric columns, scatter plots are ideal.'''
 
import pandas as pd
import matplotlib.pyplot as plt

# Create and display the first scatter plot
data.plot(kind='scatter', x='Initial Cost', y='Total Est Fee', rot=70)
plt.show()
#%%

'''Recognizing tidy data

For data to be tidy, it must have:

--Each variable as a separate column.
--Each row as a separate observation.'''


#%%

''' MELTING DATA 

Reshaping your data using melt
    -Melting data is the process of turning columns of your data into rows of data.
    In this exercise, you will practice melting a DataFrame using pd.melt(). 
    There are two parameters you should be aware of: id_vars and value_vars. 
    The id_vars represent the columns of the data you do not want to melt (i.e., keep it in its current shape),
    while the value_vars represent the columns you do wish to melt into rows. 
    By default, if no value_vars are provided, all columns not set in the id_vars will be melted. 
    This could save a bit of typing, depending on the number of columns that need to be melted.
    
    
    You can rename the variable column by specifying an argument to the "var_name" parameter, 
    and the value column by specifying an argument to the "value_name" parameter.
'''

airquality = pd.read_csv('C:\\Users\\Vivek\\Desktop\\PythonPrograms\\airquality.csv')
# Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')

#%%
''' Pivoting data

    -- Opposite of melting
    -- Turn unique values of the rows and turn them into seperate columns
    -- Used to get data from Analysis friendly shape to reporting friendly shape
    -- Or, when rows contain column suitable data(violation of Tidy Data Principle)
        -- Multiple variables are stored in same column
'''
      
'''USING THE ".pivot_table" method'''

# Print the head of airquality_melt
print(airquality_melt.head())

# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')

#%%

with pd.option_context('display.max_rows',None,'display.max_columns',None):
    print(data)
    
#%%
        
'''Resetting the index of a DataFrame'''

# Print the index of airquality_pivot
print(airquality_pivot.index)


# Reset the index of airquality_pivot: airquality_pivot_reset
airquality_pivot_reset = airquality_pivot.reset_index()

# Print the new index of airquality_pivot_reset
print(airquality_pivot_reset.index)

# Print the head of airquality_pivot_reset
print(airquality_pivot_reset.head())

#%%
import pandas as pd

tb = pd.read_csv('C:\\Users\\Vivek\\Desktop\\PythonPrograms\\tb.csv')

tb_melt = pd.melt(frame = tb, id_vars=['country','year' ])
#%%

#Create a 'gender' column by slicing the first letter of the variable column of tb_melt.
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column  by slicing the rest of the variable column of tb_melt.
tb_melt['age_group'] = tb_melt.variable.str[1:]
#%%
print(tb_melt.head())
print(tb.head())
#%%

'''Splitting a column with .split() and .get()'''


#%%
#Importing Ebola.csv file
ebola = pd.read_csv('C:\\Users\\Vivek\\Desktop\\PythonPrograms\\ebola.csv')

print(ebola.head())
#%%
'''This time, you cannot directly slice the variable by position as in the previous exercise. 
You now need to use Python's built-in string method called .split(). 
By default, this method will split a string into parts separated by a space. However, in this case you want it to split by an underscore.
 You can do this on Cases_Guinea, for example, using Cases_Guinea.split('_'), which returns the list ['Cases', 'Guinea']
 
The next challenge is to extract the first element of this list and assign it to a type variable, and the second element of the list to a country variable. 
You can accomplish this by accessing the str attribute of the column and using the .get() method to retrieve the 0 or 1 index, depending on the part you want.
'''
# Melt ebola using 'Date' and 'Day' as the id_vars, 'type_country' as the var_name, and 'counts' as the value_name.
# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')
print(ebola_melt.head(),end='\n')
print('\n')
print(ebola.head())
#%%

#Create a column called 'str_split' by splitting the 'type_country' column of ebola_melt on '_'. 
#Note that you will first have to access the str attribute of type_country before you can use .split().

ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')
print(ebola_melt.head(),end='\n')


#%%

#Create a column called 'type' by using the .get() method to retrieve index 0 of the 'str_split' column of ebola_melt.

ebola_melt['type'] = ebola_melt.str_split.str.get(0)
ebola_melt['country'] = ebola_melt.str_split.str.get(1)

print(ebola_melt.head())
#%%

''' CONCATENATING DATA'''

''' Combining rows of data'''

# Concatenate uber1, uber2, and uber3: row_concat
row_concat = pd.concat([uber1,uber2,uber3])

# Print the shape of row_concat
print(row_concat.shape)

# Print the head of row_concat
print(row_concat.head())

#%%

''' Combining columns of data   '''

'''To perform this action, you use the same pd.concat() function, but this time with the keyword argument axis=1. 
The default, axis=0, is for a row-wise concatenation.'''

# Concatenate ebola_melt and status_country column-wise: ebola_tidy
ebola_tidy = pd.concat([ebola_melt,status_country],axis=1)  #arg: axis = 1: Columns concate
                                                            #arg: axis = 0: row concate    
# Print the shape of ebola_tidy
print(ebola_tidy.shape)

# Print the head of ebola_tidy
print(ebola_tidy.head())

#%%
'''CONCATINATING MULTIPLE FILES
    -In order to use concatinating dataframes 
        --They must be in list[]
        -- can load each data individually if there are fewdatasets
        --what if there are thousands of files
        Solution-- "glob" function to find files based on a pattern
    '''
''' Globbing 
     -- Pattern Matching for filenames
     -- Wildcards: *? (wilcard is a symbol that will match any arbitrary no. of characters)
                   " * "-- type of wildcard that matches  any string
                   " *? "-- contrast with "?" , that will allow us to match 1 character
                   example:
                       *.csv-- Any csv file
                       file_?.csv-- Any single character'''

                      
#Finding files that match a pattern                   
import glob
import pandas as pd


# Write the pattern: pattern
pattern = '*.csv'
# Save all file matches: csv_files
csv_files = glob.glob(pattern)


# Print the file names
print(csv_files)

# Load the second file into a DataFrame: csv2
csv2 = pd.read_csv(csv_files[1])

# Print the head of csv2
print(csv2.head())

#%%
'''Merging data'''

# Merge the DataFrames: o2o
o2o = pd.merge(left=site, right=visited, left_on='name', right_on='site')

# Print o2o
print(o2o)

#%%

'''Converting Data Types
    -- df['column'].astype(type)
    -- eg. df['column'].astype('str'),df['column'].astype('category')'''
    
''' Categorical Data Types
    --Converting Categorical to 'category' dtype
      --Can make dataFrames smaller in memory
      --Can make them to be ustilized by other liberaries for analysis
      
    String to numeric values 
    --using pd.to_numeric(df['column'],errors = 'coerce')'''

#%%
    
#importing tips.csv

import pandas as pd

tips = pd.read_csv('C:\\Users\\Vivek\\Desktop\\PythonPrograms\\tips.csv')


print(tips.info())

#converting Data into categorial data type
tips.sex = tips.sex.astype('category')
tips.smoker  = tips.smoker.astype('category')
tips.day = tips.day.astype('category')

print(tips.info())


#converting str(object) datatype to numerical(int or float)
tips.total_bill = pd.to_numeric(tips.total_bill,errors= 'coerce')

#%%
''' USING REGULAR EXPRESSIONS TO CLEAN STRINGS
    --Most of the data involves string manipulation
      -- Because Most of the worlds data is unstructured text
    -- Also have to make string manipulations to make datasets  consistent with one another(so you can work with and combine them later)  
    
    Validate Values
    -- Monetory values representations
        -- 17 (int)
        -- $17 (int with a flaoting sign)
        -- $17.89 (a float number with 2 no. after decimal)
        -- $17.8967 (a float number with multiple no. after decimal)
        
    String Manipulations
    --Many built in liberaries for string manipulations
    --"re" liberary for regular expressions
        --used for string pattern matching
        -- Gives us a formal way of specify a pattern
        -- does it with a sequence of charaters
        
    NOTE: '$' has special meaning in regex which is - match from the end of the string
    
    Using Regular Expressions
    -- step-1: Compile the pattern
       step-2: Use the compiled to match values
       This let's use the pattern again
        '''
        
#%%
'''String parsing with regular expressions'''

#Compile a pattern that matches a phone number of the format xxx-xxx-xxxx.
#Use \d{x} to match x digits. Here you'll need to use it three times: twice to match 3 digits, and once to match 4 digits.
#Place the regular expression inside re.compile().

# Import the regular expression module
import re

# Compile the pattern: prog
prog = re.compile('\d{3}-\d{3}-\d{4}')

#Using the .match() method on prog, check whether the pattern matches the string '123-456-7890'.
# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result))
#%%

'''Extracting numerical values from strings
    Extracting numbers from strings is a common task, particularly when working with unstructured data or log files.

Say you have the following string:
    
    'the recipe calls for 6 strawberries and 2 bananas'.

It would be useful to extract the 6 and the 2 from this string to be saved for later use when comparing strawberry to banana ratios.

When using a regular expression to extract multiple numbers (or multiple pattern matches, to be exact), you can use the re.findall() function.
Dan did not discuss this in the video, but it is straightforward to use: 
You pass in a pattern and a string to re.findall(), and it will return a list of the matches.
'''
#Use the re.findall() function and pass it two arguments: the pattern, followed by the string.

import re

#Use the re.findall() function and pass it two arguments: the pattern, followed by the string.
#\d is the pattern required to find digits. This should be followed with a + so that the previous element is matched one or more times. 
#This ensures that 10 is viewed as one number and not as 1 and 0.

# FiA string of the format: A dollar sign, an arbitrary number of digits, a decimal point, 2 digits.nd the numeric values: matches
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)

#%%
'''Pattern Matching'''

#A telephone number of the format xxx-xxx-xxxx.
# Write the first pattern
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)


#A string of the format: A dollar sign, an arbitrary number of digits, a decimal point, 2 digits.
#Use \$ to match the dollar sign, \d* to match an arbitrary number of digits, \. to match the decimal point, and \d{x} to match x number of digits.
# Write the second pattern
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
print(pattern2)


#A capital letter, followed by an arbitrary number of alphanumeric characters.
#Use [A-Z] to match any capital letter followed by \w* to match an arbitrary number of alphanumeric characters.
# Write the third pattern
pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
print(pattern3)

#%%

'''USING FUNCTIONS TO CLEAN DATA'''

'''COMPLEX CLEANING
    --Cleaning step requires multiple steps
        --Extract number from a string
        --Perform transformation on extracted number
    -- Python functions
        Apply function:
            --df.apply(np.mean,axis=1) arg1: desired function, 
                                       arg2: axis:0 among column, 1 for among rows
                                       
'''
import numpy as np
# Define recode_gender()
def recode_gender(gender):

    # Return 0 if gender is 'Female'
    if gender == 'Female':
        return 0
    
    # Return 1 if gender is 'Male'    
    elif gender == 'Male':
        return 1
    
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
tips['recode'] = tips.sex.apply(recode_gender)

# Print the first five rows of tips
print(tips.head())

#%%    

'''Lambda functions
    --Instead of using the def syntax that you used in the previous exercise, lambda functions let you make simple, one-line functions.'''

#Data Column is not available hence code reference    
    
# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())
#%%

'''Duplicate data
    -- can skew results
    --".drop_duplicates()" method
    --'''
tips.drop_duplicates()

#%%

''' Missing data
    --"Nan" missing value in a dataset
    -- Varius ways to deal with missing data
        --Leave as it is
        --Drop them
        --Fill missing values using various pandas method'''
        
# Drop missing values
#if a row contains a missing value, the row is dropped.
tips_dropped = tips.dropna()

# NOTE: Dropping a column is also a possibilty if most missing values are from particular problem

#%%

'''Fill missing values using ".fillna()" method
    -- Fill it with user provided value
    -- Use some summary statistics
   
   Fill missing values with a test statistic
    -- Carefull when using test statistics to fill
    --Have to make sure that the value you are filling in makes sense
    -- Median is better statistc in the presence of outliers '''
    
    
# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality['Ozone'].mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality['Ozone'].fillna(oz_mean)

# Print the info of airquality
print(airquality.info())


#%%

'''Testing with Asserts
    Assert Statements:
        -programatically check our data vs visualy checking it
        -for eg: if drop or fill NaNs, we expect 0 missing values
        - we can write assert statements to make sure there is no more missing data
        -we can detect early warnings and results before final results
        - this gives us confidence that our code is correctly
        - help re-run analysis with updated data
'''
# Using ebola.csv dataset

# Using ".notnull()" method to check NaN values in dataset, output rows will be boolean

print(ebola.notnull())

# Using ".all()" method along with ".notnull()" method. The .all() method returns True if all values are True

print(ebola.notnull().all())

# Another way of using 

print(pd.notnull(ebola))

print(pd.notnull(ebola).all())


''' Using Assert Statements '''

#Use the pd.notnull() function on ebola (or the .notnull() method of ebola) and chain two .all() methods (that is, .all().all()).
#The first .all() method will return a True or False for each column, while the second .all() method will return a single True or False.

assert pd.notnull(ebola).all().all()

#Write an assert statement to confirm that all values in ebola are greater than or equal to 0.
#Chain two all() methods to the Boolean condition (ebola >= 0).
    
assert (ebola>=0).all().all()

#%%

''' Putting the above all learning together, Applying on Gapminder data'''
import pandas as pd

gapminder  = pd.read_csv('C:\\Users\\Vivek\\Desktop\\PythonPrograms\\gapminder.csv')

gapminder.info()



''' EXPLORATORY DATA ANALYSIS'''

'''Whenever you obtain a new dataset, your first task should always be to do some exploratory analysis to get a better understanding of the data and diagnose it for any potential issues.

The Gapminder data for the 19th century has been loaded into a DataFrame called g1800s. In the IPython Shell, use pandas methods such as .head(), .info(), and .describe(), 
and DataFrame attributes like .columns and .shape to explore it.'''


print(gapminder.head(),end='\n')

print(gapminder.info(),end='\n')

print(gapminder.shape,end='\n')

print(gapminder.describe(),end='\n')

print(gapminder.columns,end='\n')



''' VISUALIZING YOUR DATA'''

'''Here, the goal is to visually check the data for insights as well as errors. 
When looking at the plot, pay attention to whether the scatter plot takes the form of a diagonal line, and which points fall below or above the diagonal line. 
This will inform how life expectancy in 1899 changed (or did not change) compared to 1800 for different countries. 
If points fall on a diagonal line, it means that life expectancy remained the same!
'''

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create the scatter plot
plot= gapminder.plot(kind='scatter', x='1800', y='1899')

# Specify axis labels
plt.xlabel('Life Expectancy by Country in 1800')
plt.ylabel('Life Expectancy by Country in 1899')

# Specify axis limits
plt.xlim(20, 55)
plt.ylim(20, 55)

# Display the plot
plt.show()

#%%

def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
    no_na = row_data.dropna()
    numeric = pd.to_numeric(no_na)
    ge0 = numeric >= 0
    return ge0

# Check whether the first column is 'Life expectancy'
assert gapminder.columns[0] == 'Life expectancy'
# Check whether the values in the row are valid
assert gapminder.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# Check that there is only one instance of each country
assert gapminder['Life expectancy'].value_counts()[0] == 1

#%%

import pandas as pd

gapminder.drop('Unnamed: 0',axis=1)
# Melt gapminder: gapminder_melt
gapminder_melt = pd.melt(gapminder,id_vars='Life expectancy')



# Rename the columns
gapminder_melt.columns = ['country','year','life_expectancy']

# Print the head of gapminder_melt
print(gapminder_melt.head())

#%%


























#%%











