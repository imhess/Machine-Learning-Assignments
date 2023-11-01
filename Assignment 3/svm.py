#-------------------------------------------------------------------------
# AUTHOR: Isaiah Hessler
# FILENAME: svm.py
# SPECIFICATION: Complete the Python program (svm.py) that will read the file optdigits.tra (3,823 samples)
# that includes training instances of handwritten digits (optically recognized). Read the file
# optdigits.names to get detailed information about this dataset. Also, check the file optdigits-orig.tra and
# optdigits-orig.names to see the original format of this data, and how it was transformed to speed-up the
# learning process (pre-processing phase). Your goal is to build multiple SVM classifiers using this data.
# You will simulate a grid search, trying to find which combination of four SVM hyperparameters (c,
# degree, kernel, and decision_function_shape) leads you to the best prediction performance. To test the
# accuracy of those distinct models, you will use the file optdigits.tes (1,797 samples).
# FOR: CS 4210- Assignment #3
# TIME SPENT: 3 Hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('C:/Users/Isaiah/Documents/GitHub/Machine-Learning-Assignments/Assignment 3/optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('C:/Users/Isaiah/Documents/GitHub/Machine-Learning-Assignments/Assignment 3/optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape

highest_accuracy = 0

for val in c:
    for d in degree:
        for k in kernel:
           for f_s in decision_function_shape:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                clf = svm.SVC(C=val, degree=d, kernel=k, decision_function_shape=f_s)

                #Fit SVM to the training data
                clf.fit(X=X_training, y=y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                total = 0
                correct = 0
                counter = 0
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    x_pred = clf.predict([x_testSample])
                    if x_pred[0] == y_test[counter]:
                        correct += 1
                    total += 1
                    counter += 1


                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                accuracy = (correct/total)

                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    accuracy_rounded = round(highest_accuracy, 3)
                    print(f'Highest SVM accuracy so far: {accuracy_rounded}, Parameters: a = {val}, degree = {d}, kernel = {k}, decision_function_shape = {f_s}')




