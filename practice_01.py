#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:10:10 2018

@author: shymacbook
"""

#   HW01 - ID3 Decision Tree

import pandas as pd
import os
import numpy as np
import math

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

#============================================================================
#----------------------------------------------------------   import the data
print(os.getcwd())
df = pd.read_csv('/Users/shymacbook/Documents/BC/cs460_ML/HW/cs460_DecisionTree_id3/credit.csv')

#============================================================================
# ----------------------------------------------------------  check out the data
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
#----------------------------------------------------------   bucket the continuous variables
df2 = pd.read_csv('/Users/shymacbook/Documents/BC/cs460_ML/HW/cs460_DecisionTree_id3/credit.csv')
# bucket the age bins
ageBins = [18, 27, 33, 42, 100]
ageLabels = ['19-27', '27-33', '33-42', '42-100']
df2['ageBin'] = pd.cut(df['age'], bins=ageBins, labels=ageLabels)
# bucket the months_loan_duration bins
mDurBins = [0, 12, 18, 24, 100]
mDurLabels = ['0-12', '12-18', '18-24', '24-100']
df2['monDurBin'] = pd.cut(df['months_loan_duration'], bins=mDurBins, labels=mDurLabels)
# bucket the amount bins
amtBins = [1, 1360, 2320, 3970, 20000]
amtLabels = ['1-1360', '1360-2320', '2320-243970' ,'3970-20000']
df2['amtBin'] = pd.cut(df['amount'], bins=amtBins, labels=amtLabels)
# bucket the perfent of income bins
df2['percent_of_income'] = df2['percent_of_income'].astype(str)
# bucket the years at residence bins
df2['years_at_residence'] = df2['years_at_residence'].astype(str)
# bucket the existing loans bins
df2['existing_loans_count'] = df2['existing_loans_count'].astype(str)
# bucket the dependents bins
df2['dependents'] = df2['dependents'].astype(str)


#============================================================================
#----------------------------------------------------------     subset test/train data
# make a 90% mask to split the test data
mskTrain = np.random.rand(len(df2)) > 0.1
# mskTest = 1 - mskTrain
dfTrain = df2[mskTrain]
dfTest = df2[~mskTrain]
dfTrain = dfTrain.reset_index(drop=True)
dfTest = dfTest.reset_index(drop=True)
dfTrain.describe()
dfTest.describe()

#============================================================================
#----------------------------------------------------------     Define Attributes and Target
# use all columns, except for amt and age. These ones are continuous, we only want to use the discrete bins
attributes = ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'employment_duration', 'percent_of_income', 'years_at_residence', 'other_credit', 'housing', 'existing_loans_count', 'job', 'dependents', 'ageBin', 'amtBin', 'monDurBin']
target = 'default'

#============================================================================
#----------------------------------------------------------     explore data...
for x in attributes:
    print(x, ': ')
    for w in df2[x].unique():
        print('\t',w)
df2.head()



#============================================================================
#----------------------------------------------------------     Entropy Function

def entropy(df, targAttr = 'default'):
    label_freq = {}
    data_entropy = 0.0
    # put label value counts into dictionary
    for x in range(0,len(df.index)):
        if(df[targAttr][x] in label_freq):
            label_freq[df[targAttr][x]] += 1.0
        else:
            label_freq[df[targAttr][x]] = 1.0
    # calculate entropy
    for freq in label_freq.values():
        data_entropy += (-freq/len(df)) * math.log(freq/len(df), 2)
    return data_entropy

#----------------------------------------------------------     manual test of Entropy() on df2
entropy(df2)
yesCount = 0
noCount = 0
for x in range(0,len(df2.index)):
    if(df['default'][x] == 'yes'):
        yesCount += 1
    if(df['default'][x] == 'no'):
        noCount +=1
print('yes = ', yesCount, '\tno = ', noCount)
pYes = yesCount / (yesCount + noCount)
pNo = noCount / (yesCount + noCount)
manualEntropy = -(pYes * math.log2(pYes)) - (pNo * math.log2(pNo))
print('manual entropy = ', manualEntropy)


#============================================================================
#----------------------------------------------------------     Gain Function
def gain(df, attribute, targetAttr = 'default'):
    attribute_freq = {}
    subset_entropy = 0.0
    # put attribute value counts into dictionary
    for x in range(0, len(df.index)):
        if(df[attribute][x] in attribute_freq):
            attribute_freq[df[attribute][x]] += 1.0
        else:
            attribute_freq[df[attribute][x]] = 1.0
    # calculate subset entropies
    for attr in attribute_freq.keys():
        attr_prob = attribute_freq[attr] / sum(attribute_freq.values())
        subsetMask = df[attribute] == attr
        data_subset = df[subsetMask]
        data_subset = data_subset.reset_index(drop=True)
        subset_entropy += attr_prob * entropy(data_subset, targetAttr)
    # calculate gain
    return (entropy(df, targetAttr) - subset_entropy)


#============================================================================
#----------------------------------------------------------     Tree Function
def tree(df, attributes, target):
    ...
    
    

    
    
    
    


#============================================================================
#----------------------------------------------------------     getMax function
def getMaxGainAttr(df, attributes): 
    testDict = {}
    for x in attributes:
        testDict[x] = gain(df, attribute = x, targetAttr = 'default')
    maxGainAttribute = max(testDict, key=testDict.get)
    return maxGainAttribute






    
    
    
#============================================================================
#----------------------------------------------------------     testing
# printing the gains for all attributes
for x in attributes:
    # print(x, ' gain:\t', gain(dfTrain, attribute = x, targetAttr = 'default'))
    print("{: >20} {: >6} {: >10}".format(x, 'gain:', gain(dfTrain, attribute = x, targetAttr = 'default')))

getMaxGainAttr(dfTest, attributes)
    
    
    

