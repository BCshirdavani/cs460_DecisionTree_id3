#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:48:53 2018

@author: shymacbook
"""
import pandas as pd
import os
import numpy as np
import math

# practicing from book example
bookData = pd.read_csv('/Users/shymacbook/Documents/BC/cs460_ML/HW/cs460_DecisionTree_id3/bookData.csv')
bookData
bookAttributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
bookTarget = 'PlayTennis'


def entropy(df, targAttr = 'PlayTennis'):
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

entropy(bookData)   # matches .94, same value as page 56


def gain(df, attribute, targetAttr = 'PlayTennis'):
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


gain(bookData, 'Wind')  # matches .048, same value as page 58
gain(bookData, 'Outlook')
gain(bookData, 'Humidity')
gain(bookData, 'Temperature')



def getMaxGainAttr(df, attributes, target): 
    gainDict = {}
    for x in attributes:
        gainDict[x] = gain(df, attribute = x, targetAttr = target)
    maxGainAttribute = max(gainDict, key=gainDict.get)
    return maxGainAttribute


def majority(df, target = 'PlayTennis'):
    df = df.reset_index(drop=True)
    values = {'yes':0, 'no':0}              # This will only work for yes/no target values
    for x in range(0, len(df.index)):
        values[(df[target][x])] += 1.0
    maxVal = max(values, key=values.get)
    return maxVal
        

getMaxGainAttr(bookData, bookAttributes, 'PlayTennis')

def getValues(df, attribute):
    values = []
    values = df[attribute].unique()
    return values


def getExampleSubset(data, bestAttr, val):
    mask = data[bestAttr] == val
    examples = data[mask]
    examples = examples.reset_index(drop=True)
    return examples


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

bookTree = makeTree(bookData, bookAttributes, bookTarget, 0)
bookTree        # so far, this tree matches the tree on page 61






class Traverse:
    def __init__(self, dicIN):
        self.treeDic = dicIN
        self.path = 0
        self.tempList = []
        self.pathList = []
        self.rulesList = []
        self.temp = ''
        self.ruleDict = {}
        
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ERROR
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ rulesDict is fucking up
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    '''
    bookPath.ruleDict
    Out[92]: 
    {0: {'Outlook': 'Sunny', 'Humidity': 'Normal'},
     1: {'Outlook': 'Sunny', 'Humidity': 'Overcast'},
     2: {'Outlook': 'Sunny', 'Humidity': 'Rain', 'Wind': 'Weak'}}
    '''
    # Overcase and Rain, should be for Outlook, not Humidity...
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
                kvPairs = list(self.ruleDict[rule].items())  
                for pair in kvPairs:
                    print('\t\tpair = ', pair, 'actual = ', df[pair[0]][rowNum])
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

bookTree = makeTree(bookData, bookAttributes, bookTarget, 0)
bookPath = Traverse(bookTree)
bookPath.traverse(bookTree)
bookPath.ruleDict
bookPredict = bookPath.makeLabels(bookData)
bookPredict

total = len(bookPredict)
for row in range(0, len(bookPredict)):
    if newDF2['default'][row] == bookPredict['predict_label'][row]:
        correctLabel += 1
accuracy = correctLabel / total
accuracy















