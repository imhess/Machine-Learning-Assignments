#-------------------------------------------------------------------------
# AUTHOR: Isaiah Hessler
# FILENAME: Assignment 2 - Part 3
# SPECIFICATION: Complete the Python program (naïve_bayes.py) that will read the file
# weather_training.csv (training set) and output the classification of each test instance from the file
# weather_test (test set) if the classification confidence is >= 0.75.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

train = []
test = []

#reading the training data in a csv file
with open('C:\\Users\\thoma\\Documents\\GitHub\\Machine-Learning-Assignments\\Assignment 2\\Part 3\\weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         train.append(row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
X = []
for row in train:
  temp = []
  temp.clear()
  for j, value in enumerate(row):
    if j == 1:
      if value == 'Sunny':
        temp.append(1) 
      elif value == 'Overcast':
        temp.append(2)
      else:
        temp.append(3)
    if j == 2:
      if value == 'Hot':
        temp.append(1) 
      elif value == 'Mild':
        temp.append(2)
      else:
        temp.append(3)
    if j == 3:
      if value == 'High':
        temp.append(1) 
      else:
        temp.append(2)
    if j == 4:
      if value == 'Strong':
        temp.append(1) 
      else:
        temp.append(2)
  X.append(temp)

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for row in train:
  if row[-1] == 'Yes':
    Y.append(1)
  else:
    Y.append(2)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
with open('C:\\Users\\thoma\\Documents\\GitHub\\Machine-Learning-Assignments\\Assignment 2\\Part 3\\weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         test.append (row)
      else:
        heading = row

#printing the header of the solution
for label in heading:
  print(label, end='  ')

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here


