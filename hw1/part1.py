import math
import numpy as np
from collections import Counter
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Part 1: Decision Tree (with Discrete Attributes) -- 40 points --
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs:
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar
            C: the dictionary of attribute values and children nodes.
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes).
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #use a dictionary to count the number of each attribute sample
        counter = Counter()
        for value in Y:
            counter[value] += 1

        #the number of samples in Y
        sample_size = len(Y)
        #entropy
        e = 0
        for element in counter:
            e += -(math.log2(counter[element] / sample_size)) * (counter[element] / sample_size)
        #########################################
        return e



    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value.=> what does that mean
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        counter = Counter()
        dict = {}
        ce = 0
        index = 0

        for element in X:
            counter[element] += 1
            if element not in dict:
                dict[element] = []
            dict[element].append(Y[index])
            index += 1

        for key, value in dict.items():
            ce += Tree.entropy(value) * len(value) / len(X)
        #########################################
        return ce



    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Compute the information gain of y after spliting over attribute x
            InfoGain(Y,X) = H(Y) - H(Y|X)
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = Tree.entropy(Y) - Tree.conditional_entropy(Y, X)
        #########################################
        return g


    #--------------------------
    @staticmethod
    def best_attribute(X,Y):
        '''
            Find the best attribute to split the node.
            Here we use information gain to evaluate the attributes.
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        global_max = 0
        attribute_index = 0
        current_index = 0
        for att in X:
            info_gain = Tree.information_gain(Y, att)
            if info_gain > global_max:
                global_max = info_gain
                attribute_index = current_index
            current_index += 1
        #########################################
        return attribute_index


    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.

            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes.
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE#TODO
        att_dict = dict()
        label = dict()
        C = dict()
        attributes = X[i]
        index = 0

        for attribute in attributes:
            if attribute not in att_dict:
                att_dict[attribute] = []
            att_dict[attribute].append(index)
            index += 1

        for key, value in att_dict.items():
            sub_matrix = []
            for i in range(len(X)):
                sub_matrix.append([X[i][k] for k in value])
                label = [Y[j] for j in value]
                index += 1
            sub_matrix = np.reshape(sub_matrix, (len(X), len(value)))
            label = np.array(label)
            C[key] = Node(sub_matrix, label)

        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label.

            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar.
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        initial_attribute = Y[0]
        for att in Y:
            if att != initial_attribute:
                return False
        s = True
        #########################################
        return s

    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attribute values.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #check matrix dimension
        #range, and shape[0] shape[1]
        for data_sample in X:
           initial = data_sample[0]
           for data in data_sample:
               if data != initial:
                   return False
        s = True
        #########################################
        return s


    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        count_dict = {}
        for y in Y:
            if not y in count_dict:
                count_dict[y] = 1
            else:
                count_dict[y] += 1

        y = max(count_dict.keys(), key =(lambda k: count_dict[k]))
        #########################################
        return y



    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes.
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        if Tree.stop1(t.Y) or Tree.stop2(t.X):
            t.isleaf = True
            t.p = Tree.most_common(t.Y)
            return t
        attribute_index = Tree.best_attribute(t.X, t.Y)
        t.i = attribute_index
        t.p = t.Y[attribute_index]
        t.C = Tree.split(t.X, t.Y, attribute_index)
        for key, value in t.C.items():
            Tree.build_tree(value)
        return t

        #########################################

    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #create root node
        t = Node(X, Y)
        t = Tree.build_tree(t)
        #########################################
        return t



    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively.
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #Base case:
        if t.isleaf == True or t.C is None:
            return t.p
        attribute_index = t.i
        if x[attribute_index] not in t.C.keys():
            return t.p
        for key, value in t.C.items():
            if key == x[attribute_index]:
                return Tree.inference(value, x)
        #########################################


    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset.
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        transposed_X = np.transpose(X)
        Y = np.array([])
        for data_sample in transposed_X:
            label = Tree.inference(t, data_sample)
            Y = np.append(Y, label)
        #########################################
        return Y



    #--------------------------
    @staticmethod
    def load_dataset(filename = 'data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'.
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #
        f = open('data1.csv', "r")
        X = np.array([])
        Y = np.array([])
        lines = f.readlines()
        sum = 0
        for line in lines:
            sum += 1
        line_number = sum - 1

        for i in range(len(lines)):
            if i == 0:
                attribute_number = len(lines[i].split(",")[1:])
                X = [[0 for x in range(line_number)] for y in range(attribute_number)]
            else:
                Y = np.append(Y, lines[i].split(",")[0])
                row = lines[i].split("\n")[0].split(",")[1:]
                for attribute_index in range(attribute_number):
                    X[attribute_index][i - 1] = row[attribute_index]
        X = np.array(X)

        #########################################
        return X,Y
