def makeTree(data, attributes, target, recursion):
    recursion += 1
    #Returns a new decision tree based on the examples given.
    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = majority(attributes, data, target)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = chooseAttr(data, attributes, target)
        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best:{}}
    
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in getValues(data, attributes, best):
            # Create a subtree for the current value under the "best" field
            examples = getExamples(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = makeTree(examples, newAttr, target, recursion)
    
            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree
    
    return tree








# dfTest01 = pd.DataFrame([('bird',    389.0),('bird',     24.0),('mammal',   80.5),('mammal', np.nan)],index=['falcon', 'parrot', 'lion', 'monkey'], columns=('class', 'max_speed'))
# dfTest01.head()
# dfTest01.reset_index(drop=True)
# dfTest01 = dfTest01.reset_index(drop=True)
# dfTest01.head()

for each in dfTest.purpose.unique():
    print(each)
    
    
    
    
    
dictTest = {}
dictTest[1] += ' ' + 'smee'
dictTest
    
    





class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)
        
        
        
        
        
#find most common value for an attribute
def majority(attributes, data, target):
    #find target attribute
    valFreq = {}
    #find target in data
    index = attributes.index(target)
    #calculate frequency of values in target attr
    for tuple in data:
        if (valFreq.has_key(tuple[index])):
            valFreq[tuple[index]] += 1 
        else:
            valFreq[tuple[index]] = 1
    max = 0
    major = ""
    for key in valFreq.keys():
        if valFreq[key]>max:
            max = valFreq[key]
            major = key
    return major
        
        
        
        
        
        
        
        
        
        
    #get values in the column of the given attribute 
def getValues(data, attributes, attr):
    index = attributes.index(attr)
    values = []
    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])
    return values





def getExampleSubset(data, attributes, bestAttr, val):
    mask = data[bestAttr] == val
    examples = data[mask]
    examples.reset_index(drop=True)
    return examples




attributes
len(attributes) 
attrTest = attributes[:]
attrTest.remove('monDurBin')    
attrTest    
len(attrTest)
len(attributes)
attributes


attributes[0]

#============================================================================
#----------------------------------------------------------     testing
# printing the gains for all attributes
for x in attributes:
    # print(x, ' gain:\t', gain(dfTrain, attribute = x, targetAttr = 'default'))
    print("{: >20} {: >6} {: >10}".format(x, 'gain:', gain(dfTrain, attribute = x, targetAttr = 'default')))

getMaxGainAttr(dfTest, attributes)
    


    
def makeTree(data, attributes, target, recursion):
    recursion += 1
    #Returns a new decision tree based on the examples given.
    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = majority(attributes, data, target)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 4:01pm
        best = chooseAttr(data, attributes, target)
        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best:{}}
    
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in getValues(data, attributes, best):
            # Create a subtree for the current value under the "best" field
            examples = getExamples(data, attributes, best, val) #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 4:08pm
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = makeTree(examples, newAttr, target, recursion)
    
            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree
    
    return tree










