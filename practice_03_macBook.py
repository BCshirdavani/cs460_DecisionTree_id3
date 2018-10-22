#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:31:32 2018

@author: shimac
"""

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
# MacBook
df = pd.read_csv('/Users/shymacbook/Documents/BC/cs460_ML/HW/cs460_DecisionTree_id3/credit.csv')
# iMac
# df = pd.read_csv('/Users/shimac/Documents/ComputerSci/cs460_ML/hw01/cs460_DecisionTree_id3/credit.csv')

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
# iMac
# df2 = pd.read_csv('/Users/shimac/Documents/ComputerSci/cs460_ML/hw01/cs460_DecisionTree_id3/credit.csv')
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
#----------------------------------------------------------     getMax function
def getMaxGainAttr(df, attributes, target): 
    gainDict = {}
    for x in attributes:
        gainDict[x] = gain(df, attribute = x, targetAttr = target)
    maxGainAttribute = max(gainDict, key=gainDict.get)
    return maxGainAttribute


#============================================================================
#----------------------------------------------------------     return majority value method
def majority(df, target = 'default'):
    df = df.reset_index(drop=True)
    values = {'yes':0, 'no':0}                                  # This will only work for yes/no target values
    for x in range(0, len(df.index)):
        values[(df[target][x])] += 1.0
    maxVal = max(values, key=values.get)
    return maxVal
        


#============================================================================
#----------------------------------------------------------     getValues method
def getValues(df, attribute):
    values = []
    values = df[attribute].unique()
    return values



#============================================================================
#----------------------------------------------------------     getExamples method
def getExampleSubset(data, bestAttr, val):
    mask = data[bestAttr] == val
    examples = data[mask]
    examples = examples.reset_index(drop=True)
    return examples





#============================================================================
#----------------------------------------------------------     define Node for tree
class Node:
    value = ""
    children = []
    
    def __init__(self, val, dictionary):
        self.setValue(val)
        self.genChildren(dictionary)
    
    def setValue(self, val):
        self.value = val
        
    def genChildren(self, dictionary):
        if(isinstance(dictionary, dict)):
            self.children = dictionary.keys()
        




#============================================================================
#----------------------------------------------------------     Tree Function
    

def makeTree(data, attributes, target, recursion):
    # print(data[target].unique())
    recursion += 1
    data = data[:]
    default = majority(data, target)
    # id dataset is empty, or attributes list is empty -> return default
    if ((len(data.index)) <=0) or ((len(attributes)) <= 0):
        print('~~~~~~~~~~~~~~~~~~~~ LEAF: no more data or attributes ~~~~~~~~~~~~~~ 0')
        return default
    # if all records in subset show the same classification, return that label
    # checking if only 1 unique value exists in target col
    elif (len(data[target].unique())) <= 1:
        print('~~~~~~~~~~~~~~~~~~~~ LEAF: only 1 unique default value ~~~~~~~~~~~~~~ 1')
        onlyValue = data[target].unique()[0]
        return onlyValue
    else:
        print('~~~~~~~~~~~~~~~~~~~~ else: tree recursion ~~~~~~~~~~~~~~ 2')
        # choose next best attribute to label data
        # best = getMaxGainAttr(data, target)                                     # target = 'default'
        best = getMaxGainAttr(data, attributes, target)  
        # create new tree with best attribute as root
        tree = {best:{}}
        # create subtree for each value in best attribute field
        for val in getValues(data, best):
            # create subtree for this current value
            examples = getExampleSubset(data, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = makeTree(examples, newAttr, target, recursion)
            # add the new subtree to the empty dictionary with root from earlier
            tree[best][val] = subtree
    return tree




myTree = makeTree(dfTest, attributes, 'default', 0)

print(myTree)

# practice dictionary operations for the tree traversal
len(list(myTree.keys()))
list(myTree.keys())[0]
myTree[list(myTree.keys())[0]].keys()
len(list(myTree[list(myTree.keys())[0]].keys()))
list(myTree[list(myTree.keys())[0]].keys())[0]


myTree.keys()                                                       # root key - root name
myTree
myTree[list(myTree.keys())[0]]                                      # root dict
list(myTree[list(myTree.keys())[0]].keys())                         # root leaves
atRoot = myTree[list(myTree.keys())[0]]  
atRoot.keys()
list(atRoot.keys())
list(atRoot.keys())[0]
rootChildren = list(atRoot.keys())
rootChildren
atRoot[rootChildren[0]]                                             # left child of root ('1-200' checkin_balance)
atRoot[rootChildren[0]].get('ageBin')
atRoot[rootChildren[0]].get('ageBin').get('33-42')
atRoot[rootChildren[0]].get('ageBin').get('33-42').get('percent_of_income')
print(myTree.keys())


# applying an if statement of the tree to a data frame, to mask the 'no' values of a section from the model
testNoMask = (dfTest['checking_balance'] == '1 - 200 DM') & (dfTest['ageBin'] == '33-42') & (dfTest['percent_of_income'] == '2') & (dfTest['credit_history'] == 'good') & (dfTest['purpose'] == 'car') & (dfTest['savings_balance'] == '< 100 DM') & (dfTest['employment_duration'] == '1 - 4 years') & (dfTest['years_at_residence'] == '2') & (dfTest['other_credit'] == 'none') & (dfTest['housing'] == 'rent') & (dfTest['existing_loans_count'] == '1') & (dfTest['job'] == 'management') & (dfTest['dependents'] == '1') & (dfTest['amtBin'] == '3970-20000') & (dfTest['monDurBin'] == '24-100')
dfTest[testNoMask]['default']      # correctly shows no for this 1 instance
testNoMask.sum()

# tree algo should have stopped recursion after the percent_of_income attribute....
testNoMaskShorter = (dfTest['checking_balance'] == '1 - 200 DM') & (dfTest['ageBin'] == '33-42') & (dfTest['percent_of_income'] == '2') 
dfTest[testNoMaskShorter]['default']      # correctly shows no for this 1 instance
dfTest[testNoMaskShorter]
testNoMaskShorter.sum()


#======================================================================= Class Start     
class Traverse:
    def __init__(self, dicIN):
        self.treeDic = dicIN
        self.path = 0
        self.tempList = []
        self.pathList = []
        self.rulesList = []
        self.temp = ''
        self.ruleDict = {}
        
    def traverse(self, tree):
        print('~~~ tree keys: ', tree.keys())
        children = list(tree.keys())
        for x in range(0, len(children)):
            print('\t~~~ x = ', x)
            if children != None:
                if (tree.get(children[x]) != 'yes') & (tree.get(children[x]) != 'no'):
                    self.tempList.append(children[x])                  # append this child to the ifList
                    # traverse(tree = tree.get(children[x]))             # sub tree traversal
                    self.traverse(tree = tree.get(children[x])) 
                elif (tree.get(children[x]) == 'yes'):
                    print('~~~ yes found...')
                    self.tempList.append(children[x])       # NEW
                    tempString = ''
                    i = 0
                    size = len(self.tempList)
                    for tok in self.tempList:
                        if i % 2 == 0:
                            self.temp = tok
                            i += 1
                        elif i % 2 == 1:
                            if i == 1:
                                self.ruleDict[self.path] = {self.temp : tok}
                                i += 1
                            else:
                                self.ruleDict[self.path].update({self.temp : tok})
                                i += 1
                    self.pathList.append(tempString)                    # save this path in the list of permanent paths
                    self.path += 1                                      # +1 to path counter

                    self.tempList.pop() 
                elif (tree.get(children[x]) == 'no'):
                    print('~~~~leaf at no...')
        print('\n~*~*~*~*~* RETURNING LIST ~*~*~*~')

    # method returns a new data frame with predicted label column
    def makeLabels(self, data):
        # initialize a new predicted label col to be all 'no'
        df = data[:]
        df['predict_label'] = 'no'
        rules = list(self.ruleDict.keys())
        # for each row, apply every rule
        for rowNum in range(0, len(df)):
            print('at row', rowNum)
            ruleMatch = False
            matchCount = 0
            for rule in rules:
                print('\tmatches:', matchCount, 'of', len(self.ruleDict[rule].keys()), 'ruleMatch=', ruleMatch)
                kvPairs = list(self.ruleDict[rule].items()) # < < < < < < < < < < < < < < TYPO fixed.... changed testPath. to self.
                for pair in kvPairs:
                    testAttr = pair[0]
                    testVal = pair[1]
                    if df[testAttr][rowNum] != testVal:
                        ruleMatch = False
                    elif df[testAttr][rowNum] == testVal:
                        matchCount += 1
                if matchCount == len(self.ruleDict[rule].keys()):
                    ruleMatch = True
                    print('\t\ttrue match')
                    df['predict_label'][rowNum] = 'yes'
            if ruleMatch == True:                       # redundant section...remove
                print('\t\ttrue match')                 # redundant section...remove
                df['predict_label'][rowNum] = 'yes'     # redundant section...remove
        return df

        
        
#======================================================================= Class End
myTree
testPath = Traverse(myTree)
testPath.traverse(myTree)   
testPath.ruleDict
testPath.ruleDict[0]
testPath.ruleDict[1]
testPath.ruleDict[2]
len(testPath.ruleDict[0].keys())

newDF = testPath.makeLabels(dfTest)             # row 98 shows a match...3 of 3 conditions for one of the rules...
newDF.tail()






#==========================================================================
# parsing and displaying the 'if attribute =' path for each yes leaf
testPath.ruleDict[0].items()
x = list(testPath.ruleDict[0].items())
x
x[0][0]
x[0][1]
x[1]
len(x)

testPath.ruleDict.keys()
len(testPath.ruleDict.keys())
rules = list(testPath.ruleDict.keys())
rules
for rule in rules:
    print('path', rule, ':')
    kvPairs = list(testPath.ruleDict[rule].items())
    for pair in kvPairs:
        print('if', pair[0], ' = ', pair[1])


#========================================================================== TESTING
myTree2 = makeTree(dfTrain, attributes, 'default', 0)
testPath2 = Traverse(myTree2)
testPath2.traverse(myTree2)   
testPath2.ruleDict
newDF2 = testPath2.makeLabels(dfTrain) 
newDF2.head()
newDF2.tail()

# accuracy against dfTest
yesCount = 0
noCount = 0
predYes = 0
predNo = 0
correctLabel = 0
total = len(newDF2)
for row in range(0, len(newDF2)):
    if newDF2['default'][row] == newDF2['predict_label'][row]:
        correctLabel += 1
accuracy = correctLabel / total
accuracy


#========================================================================== TESTING book example
bookData = pd.read_csv('/Users/shymacbook/Documents/BC/cs460_ML/HW/cs460_DecisionTree_id3/bookData.csv')
bookData
bookAttributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
bookTarget = 'PlayTennis'
bookTree = makeTree(dfTrain, attributes, 'PlayTennis', 0)
