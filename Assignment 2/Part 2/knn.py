#-------------------------------------------------------------------------
# AUTHOR: Isaiah Hessler
# FILENAME: Assignment 2 - Part 2
# SPECIFICATION: Complete the Python program (knn.py) that will read the file binary_points.csv
# and output the LOO-CV error rate for 1NN
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 Hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#loop your data to allow each instance to be your test set
counter = 0
right = 0
wrong = 0
error_rate = 0
for feature in db:

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    X = []
    X.clear()
    temp = []
    temp.clear()
    for i in range(len(db)):
        if feature != db[i]:
          temp.append(float(db[i][0]))
          temp.append(float(db[i][1]))
          X.append(temp[:])
          temp.clear()
        else:
           counter += 1

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    Y = []
    for j in range(len(db)):
        if feature != db[j]:
          Y.append(float(1)) if db[j][2] == '+' else Y.append(float(2))
        else:
           counter += 1
    

    #store the test sample of this iteration in the vector testSample
    testSample = []
    testSample.append(float(feature[0]))
    testSample.append(float(feature[1]))

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    true_label = 1.0 if feature[2] == '+' else 2.0
    if class_predicted == true_label:
       right += 1
    else:
       wrong += 1

#print the error rate
error_rate = (wrong / (right + wrong)) * 100
print(f'Error rate is: {error_rate}%')






