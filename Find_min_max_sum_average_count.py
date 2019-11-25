# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:37:45 2019

@author: vikash
"""
import pandas as pd
filename= 'Data Set\IPL\matches.csv'
df=pd.read_csv(filename)



print('Maximum : ', df['win_by_runs'].max())
print('Minimum : ', df['win_by_runs'].min())
print('Mean : ', df['win_by_runs'].mean())
print('Sum : ', df['win_by_runs'].sum())
print('Count : ', df['win_by_runs'].count())
print("Print column names and rows(Information) in each column :\n",df.count())

"""passing column name"""
column_name = input('Please enter column name (like win_by_runs / win_by_wickets ): ')
column_name = df[column_name]
print("User Input result")
print('Minimum : ', column_name.max())
print('Maximum : ', column_name.min())
print('Mean : ', column_name.mean())
print('Sum : ', column_name.sum())
print('Count : ', column_name.count())


"""Finding Unique Values"""
filename= 'Data Set\IPL\matches.csv'
df=pd.read_csv(filename)
column_name.unique()

"""Find unique value count"""
df['team1'].value_counts()

"""Simply count the number of unique value"""
print("Unique total team : ",df['team1'].nunique())

