#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:10:10 2018

@author: shymacbook
"""

#   HW01 - ID3 Decision Tree

import pandas as pd
import os

#============================================================================
#   import the data
print(os.getcwd())
df = pd.read_csv('/Users/shymacbook/Documents/BC/cs460_ML/HW/cs460_DecisionTree_id3/credit.csv')

#============================================================================
#   check out the data
pd.set_option('display.max_columns', None)
df.head()
df.describe().transpose()
df.hist()
contCols = ['months_loan_duration', 'amount', 'percent_of_income', 'years_at_residence', 'age', 'existing_loans_count', 'dependents']
for x in contCols:
    print(x,":")
    print(df[x].unique())
    print("\t0%: ", df[x].quantile(0), "\t25%: ", df[x].quantile(.25), "\t50%: ", df[x].quantile(.50), "\t75%: ", df[x].quantile(.75), "\t100%: ", df[x].quantile(1))

df.age.unique()

#============================================================================
#   bucket the continuous variables
df2 = pd.read_csv('/Users/shymacbook/Documents/BC/cs460_ML/HW/cs460_DecisionTree_id3/credit.csv')
# bucket the age bins
ageBins = [18, 27, 33, 42, 100]
ageLabels = ['19-27', '27-33', '33-42', '42-100']
df2['ageBin'] = pd.cut(df['age'], bins=ageBins, labels=ageLabels)
# df2['age'] = df2['age'].astype(str)
# bucket the months_loan_duration bins
# monLoanBins = [1, 12, 18, 24, 100]
# monLoanLabels = ['1-12', '12-18', '18-24', '24-100']
# df2['monthsLoanBin'] = pd.cut(df['months_loan_duration'], bins=monLoanBins, labels=monLoanLabels)
df2['months_loan_duration'] = df2['months_loan_duration'].astype(str)
# bucket the amount bins
amtBins = [1, 1360, 2320, 3970, 20000]
amtLabels = ['1-1360', '1360-2320', '2320-243970' ,'3970-20000']
df2['amtBin'] = pd.cut(df['amount'], bins=amtBins, labels=amtLabels)
# bucket the perfent of income bins
# incPctBins = [0.5, 1.5, 2.5, 3.5, 4.5]
# incPctLabels = ['0.5-1.5', '1.5-2.5', '2.5-3.5', '3.5-4.5']
# df2['incPctBins'] = pd.cut(df['percent_of_income'], bins=incPctBins, labels=incPctLabels)
df2['percent_of_income'] = df2['percent_of_income'].astype(str)
# bucket the years at residence bins
# yrResBins = [0.5, 1.5, 2.5, 3.5, 4.5]
# yrResLabels = ['0.5-1.5', '1.5-2.5', '2.5-3.5', '3.5-4.5']
# df2['yrResBins'] = pd.cut(df['years_at_residence'], bins=yrResBins, labels=yrResLabels)
df2['years_at_residence'] = df2['years_at_residence'].astype(str)
# bucket the existing loans bins
# xLoansBins = [0.5, 1.5, 2.5, 3.5, 4.5]
# xLoansLabels = ['0.5-1.5', '1.5-2.5', '2.5-3.5', '3.5-4.5']
# df2['xLoansBins'] = pd.cut(df['existing_loans_count'], bins=xLoansBins, labels=xLoansLabels)
df2['existing_loans_count'] = df2['existing_loans_count'].astype(str)
# bucket the dependents bins
df2['dependents'] = df2['dependents'].astype(str)





