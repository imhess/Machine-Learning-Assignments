#-------------------------------------------------------------------------
# AUTHOR: Isaiah Hessler
# FILENAME: Assignment 2 - Part 1
# SPECIFICATION: Complete the Python program (decision_tree_2.py) that will read the files 
# contact_lens_training_1.csv, contact_lens_training_2.csv, and contact_lens_training_3.csv. Each of
# those training sets has a different number of instances. You will observe that now the trees are being
# created setting the parameter max_depth = 3, which it is used to define the maximum depth of the tree
# (pre-pruning strategy) in sklearn. Your goal is to train, test, and output the performance of the 3 models
# created by using each training set on the test set provided (contact_lens_test.csv). You must repeat
# this process 10 times (train and test by using a different training set), choosing the average accuracy
# as the final classification performance of each model.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 Hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
testData = 'contact_lens_test.csv'

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []
    XTest = []
    YTest = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for row in dbTraining:
        temp = []
        for j, value in enumerate(row):
            print(j)
            if j == 0:
                if value == 'Young':
                    temp.append(1)
                elif value == 'Prepresbyopic':
                    temp.append(2)
                else:
                    temp.append(3)
            if j == 1:
                if value == 'Myope':
                    temp.append(1)
                else:
                    temp.append(2)
            if j == 2:
                if value == 'No':
                    temp.append(1)
                else:
                    temp.append(2)
            if j == 3:
                if value == 'Reduced':
                    temp.append(1)
                else:
                    temp.append(2)
        X.append(temp)
    print(X)

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for row in dbTraining:
        if row[-1] == 'Yes':
            Y.append(1)
        else:
            Y.append(2)

    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       dbTest = []
       with open(testData, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)

       #for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            for row in dbTraining:
                temp = []
                for j, value in enumerate(row):
                    print(j)
                    if j == 0:
                        if value == 'Young':
                            temp.append(1)
                        elif value == 'Prepresbyopic':
                            temp.append(2)
                        else:
                            temp.append(3)
                    if j == 1:
                        if value == 'Myope':
                            temp.append(1)
                        else:
                            temp.append(2)
                    if j == 2:
                        if value == 'No':
                            temp.append(1)
                        else:
                            temp.append(2)
                    if j == 3:
                        if value == 'Reduced':
                            temp.append(1)
                        else:
                            temp.append(2)
                XTest.append(temp)
            print(XTest)

            for row in dbTraining:
                if row[-1] == 'Yes':
                    YTest.append(1)
                else:
                    YTest.append(2)

           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here

    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here




