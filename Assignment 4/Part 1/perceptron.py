#-------------------------------------------------------------------------
# AUTHOR: Isaiah Hessler
# FILENAME: perceptron.py
# SPECIFICATION: Complete the Python program (perceptron.py) that will read the file optdigits.tra to build a
# Single Layer Perceptron and a Multi-Layer Perceptron classifiers. You will compare their performances
# and test which combination of two hyperparameters (learning rate and shuffle) leads you to the best
# prediction performance for each classifier. To test the accuracy of those distinct models, you will use
# the file optdigits.tes. You should update and print the accuracy of each classifier, together with the
# hyperparameters when it is getting higher.
# FOR: CS 4210- Assignment #4
# TIME SPENT: 3 Hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

algo_list = {Perceptron, MLPClassifier}

highest_accuracyM = 0
highest_accuracyP = 0

for val in n: #iterates over n

    for boo in r: #iterates over r

        #iterate over both algorithms
        for algo in algo_list: #iterates over the algorithms

            #Create a Neural Network classifier
            if algo == Perceptron:
               clf = Perceptron(eta0 = val, shuffle = boo, max_iter=1000)    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            else:
               clf = MLPClassifier(activation = 'logistic', learning_rate_init = val, hidden_layer_sizes = (25), shuffle = boo, max_iter = 1000) #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Pyhton code here

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
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

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"

            if algo == Perceptron:
               accuracyP = (correct/total)
               if accuracyP > highest_accuracyP:
                  highest_accuracyP = accuracyP
                  accuracy_roundedP = round(highest_accuracyP, 3)
                  print(f'Highest Perceptron accuracy so far: {accuracy_roundedP}, Parameters: learning rate = {n}, shuffle = {boo}')
            else:
               accuracyM = (correct/total)
               if accuracyM > highest_accuracyM:
                  highest_accuracyM = accuracyM
                  accuracy_roundedM = round(highest_accuracyM, 3)
                  print(f'Highest SVM accuracy so far: {accuracy_roundedM}, Parameters: learning rate = {n}, shuffle = {boo}')











