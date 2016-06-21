# C4.5 Decision tree
# Author: Eline Saarloos
# 2541114
# Machine Learning
# Vrije Universiteit
# 30/03/16


from __future__ import division
import numpy as np
import random
import math
from Node import make_node
from operator import itemgetter
from tree_node import tree_node
import os
import psutil
import time

start = time.time()
setosa_set = []
versicolor_set = []
virginica_set = []
training_set = []
test_set = []
sequenceOfSplits = [] #countains several lists with arguments 
current_depth = 0
current_node = 'nothing yet'

tree_from_training = [] #always exist of only one node, that node contains info about the child nodes

data = np.genfromtxt('iris.data', delimiter=',', dtype=None) # import dataset



def calculate_IG(set1, set2):
    entropysubsets = entropy_per_split(set1, set2) #calculate entropy for each subset.
    set1 += set2 #combined now so set1 is now the entire set of the node we are now in.
    entropynode = entropy_per_set(set1) #calculate entropy for entire set of a node
    gain = entropynode - entropysubsets
    return gain

def entropy_per_split(subset1, subset2): 
    entropyset1 = entropy_per_set(subset1)
    entropyset2 = entropy_per_set(subset2)
    p1 = len(subset1) / (len(subset1)+len(subset2)) #probability of set1 / total items in whole node set
    p2 = len(subset2) / (len(subset1)+len(subset2))
    entropy = (p1 * entropyset1) + (p2 * entropyset2) #Calculate the entropy: the irregularity of the data
    return entropy
    
def entropy_per_set(set): 
    setosa = 0.01 # set at a very small value since math.log does not take values of 0
    versicolor = 0.01
    virginica = 0.01
    for i in set:
        if i == 'Iris-setosa': 
            setosa += 1
        elif i == 'Iris-versicolor':
            versicolor +=1
        elif i == 'Iris-virginica':
            virginica +=1
    flowersInSet = len(set)
    p1 = setosa / flowersInSet 
    p2 = versicolor/flowersInSet
    p3 = virginica/flowersInSet
    entropy = -(p1*math.log(p1,2))-(p2*math.log(p2,2))-(p3*math.log(p3,2))
    return entropy

    

def data_to_sets():
    global training_set, test_set
    #extend the training set
    training_set.extend(setosa_set[0:35])
    training_set.extend(versicolor_set[0:35])
    training_set.extend(virginica_set[0:35])
    #extend the test set
    test_set.extend(setosa_set[35:50])
    test_set.extend(versicolor_set[35:50])
    test_set.extend(virginica_set[35:50])

def shuffle_data(): #shuffles each set
    global setosa_set, versicolor_set, virginica_set
    random.shuffle(setosa_set)
    random.shuffle(versicolor_set)
    random.shuffle(virginica_set)
    data_to_sets()

def split_data(): #seperates the data into the three Iris kinds
    global setosa_set, versicolor_set, virginica_set
    for i in data:
        if i[-1] == 'Iris-setosa':
            setosa_set.append(i)
        elif i[-1] == 'Iris-versicolor':
            versicolor_set.append(i)
        elif i[-1] == 'Iris-virginica':
            virginica_set.append(i)
        else:
            print ('Something went wrong. The datafile is invalid.') #Algorithm is specificly build for classifying the iris.data datafile.
    shuffle_data()

def split_sets(threshold, set):
    set1 = [] #here you split the sets in 2 subsets divided by the threshold
    set2 = []
    for tuple in set:
        if tuple[0] <= threshold:
            set1.append(tuple[1])
        if tuple[0] > threshold:
            set2.append(tuple[1])
    if len(set1) != 0 and len(set2) != 0: #make sure you only do for NONempty lists
        gain = calculate_IG(set1, set2) #for each subset calcuate the entropy
        return gain
        
def expand_tree_from_training(first_split, checker, position):
    global current_node
    attribute = first_split[2] #looks at what it should split.
    threshold = first_split[1]
    depth = first_split[3]
    if current_node == 'nothing yet':
        current_node = tree_node(None, None, None, threshold, attribute, depth, None) #tree_node is a class for nodes that contain information on the structure of the final learned C4.5 decision tree classifier.
        tree_from_training.append(current_node)  #tree_node
    if position == 'left':
        new_node = tree_node(current_node, None, None, threshold, attribute, depth, None)
        current_node.add_left(new_node)
        current_node = new_node
    if position == 'right':
        new_node = tree_node(current_node, None, None, threshold, attribute, depth, None)
        current_node.add_right(new_node)
        current_node = new_node

def calculate_IG_list(node, current_depth, position, checker):
    listIGs = [] #empty list because for every node you start over
    global sequenceOfSplits #SequenceOfSplits save all the best splits the tree is build with
    counter = 0 
    current_attribute = 'nothing' 
    for i in node.list_of_attribute_lists: #A list with in there 4 lists for each attribute, sorted in ascending order on value. 
        if counter == 0:
            current_attribute = 'sepal_length'
            set = node.sepal_length_and_flower #this set consists of the attribute value + the name of the flower, because the algorithm needs to know how many flowers of each type in calculating the entropy. 
        if counter == 1:
            current_attribute = 'sepal_width'
            set = node.sepal_width_and_flower
        if counter == 2:
            current_attribute = 'petal_length'
            set = node.petal_length_and_flower
        if counter == 3:
            current_attribute = 'petal_width'
            set = node.petal_width_and_flower
        for value in i: #for each value in list sorted per attributes
            threshold = value
            thesplitGain = split_sets(threshold, set) #do the split for each of the thresholds and add this to listIG's below. 
            listIGs.append([thesplitGain,threshold,current_attribute, current_depth, position])  
        counter += 1
    sortedIGs = sorted(listIGs, key=lambda tup: tup[0], reverse = True) #for all the possible split that were just calculated, sort the list based on highest IG. Highest IG is the output of thesplit and this is the first [0] item in the tuples in the listIGs.
    
    first_split = sortedIGs[0]
    expand_tree_from_training(first_split, checker, position) #this is were the first part of the learned tree is build
    
    sequenceOfSplits.append(sortedIGs[0]) 
    split_in_nodes(node, sortedIGs[0]) #Node contains the entire set of data for that node
    
    
def split_in_nodes(node, best_split):
    attribute = best_split[2] 
    threshold = best_split[1] 
    childnode1 = [] #we do a binary split so 2 childnodes, empty at first. 
    childnode2 = []
    childnodes_together = []
    if attribute == 'sepal_length':
        for i in node.set:
            if float(i[0]) <= float(threshold):
                childnode1.append(i)
            else:
                childnode2.append(i)
    if attribute == 'sepal_width':
        for i in node.set:
            if float(i[1]) <= float(threshold):
                childnode1.append(i)
            else:
                childnode2.append(i)
    if attribute == 'petal_length':
        for i in node.set:
            if float(i[2]) <= float(threshold):
                childnode1.append(i)
            else:
                childnode2.append(i)
    if attribute == 'petal_width':
        for i in node.set:
            if float(i[3]) <= float(threshold):
                childnode1.append(i)
            else:
                childnode2.append(i) 
    childnodes_together.append(childnode1)
    childnodes_together.append(childnode2)
    
    recursion(childnodes_together, node, best_split) #keeps repeating the progress until it has learned to correctly classify all data-items from the training set
                

        
def recursion(childnodes_together, parentnode, best_split):
    counter = 0
    attribute = best_split[2] 
    threshold = best_split[1]
    depth = best_split[3]
    output = 'nothing yet'
    global current_depth
    global current_node
    for i in childnodes_together:
        counter += 1
        if checkEqual(i): #check if all items in the node belong to the same category
            for x in i:
                output = x[4]
                break
            if counter == 1:
                new_child = tree_node(current_node, None, None, None, None, depth, output)
                current_node.add_left(new_child)
            if counter == 2:
                new_child = tree_node(current_node, None, None, None, None, depth, output)
                current_node.add_right(new_child)
        elif not checkEqual(i):
            if counter == 1:
                position = 'left'
            else:
                position = 'right'
            newchild = make_node(i, parentnode, position) #make a newnode in final learned tree
            parentnode.add_child(newchild) 
       
    checker = 1
    for child in parentnode.children:
        save_depth = current_depth
        current_depth += 1
        position = child.position
        save_current_node = current_node
        calculate_IG_list(child, current_depth, position, checker)
        current_node = save_current_node
        checker += 1
        current_depth = save_depth
        
        

def checkEqual(childnode): #check if all items in the node belong to the same category
    first_item = 0
    for x in childnode:
        first_item = x[4] 
        break
    if all(first_item == x[4] for x in childnode):
        return True
    else: 
        return False
    
           

        
split_data() #Split de data in train&testset
root = make_node(training_set, None, None) #create a rootnode that consist of data from the entire training_set
calculate_IG_list(root, current_depth, 'root', None) #startingpoint for the learning of the final classifier tree!


correctlyClassified = 0
correctSetosa = 0
correctVirginica = 0
correctVersicolor = 0
missClassified = 0


def start_testing(test_set, tree_from_training): 
#After classifier tree is build, it's performance on the test set is tested. 
    global correctlyClassified
    for i in test_set:
        walk_tree(i, tree_from_training)
    percentageVersicolor = (correctVersicolor/15)*100
    percentageVirginica = (correctVirginica/15)*100
    percentageSetosa = (correctSetosa/15)*100
    print 'Total amount of items in test set: %r' %len(test_set)
    print 'Amount of correctly classified items: %r' %correctlyClassified
    percentage = (correctlyClassified / len(test_set))
    g = float("{0:.3f}".format(percentage))
    return g

def walk_tree(i, tree): #process of data-item passing through the tree
    global correctlyClassified
    global missClassified
    global correctSetosa
    global correctVirginica
    global correctVersicolor
    target = i[4]
    for node in tree:
        atr = check_attribute(node) #checks the split attribute of the leave(node)
        o = check_output(node) #check the classification category of the leave(node)
        if target == o:
            correctlyClassified += 1 
            if target == 'Iris-setosa':
                correctSetosa +=1 
            if target == 'Iris-virginica':
                correctVirginica +=1
            if  target == 'Iris-versicolor':
                correctVersicolor += 1
            break
        if target != o and atr == None: #if target output of data item does not match category of the node
            missClassified += 1 
            break
        else:
            threshold = node.threshold
            if i[atr] <= float(threshold): 
                tree = node.leftchildren #continue the classification process in the left branch
                walk_tree(i, tree)
            if i[atr] > float(threshold):
                tree = node.rightchildren #continue the classification processin the right branch
                walk_tree(i, tree)
                
                              
            
def check_output(node):
    if node.output == 'Iris-setosa':
        return 'Iris-setosa'
    if node.output == 'Iris-virginica':
        return 'Iris-virginica'
    if node.output == 'Iris-versicolor':
        return 'Iris-versicolor'
    else:
        return False

            
def check_attribute(node):
    if node.attribute == 'sepal_length': 
        return 0
    if node.attribute == 'sepal_width':
        return 1
    if node.attribute == 'petal_length':
        return 2
    if node.attribute == 'petal_width':
        return 3

#The code below was used for collecting the C4.5 algorithm performance information.     

start_testing(test_set, tree_from_training)
#
#process = psutil.Process(os.getpid())
#outcome = process.memory_info().rss
#
#end = time.time()
#time = end - start
#    
#output = open("time.txt",'a') #opens file with name of "output.txt"
#output.write(repr(time)) 
#output.write('\n')

