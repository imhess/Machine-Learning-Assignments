#-------------------------------------------------------------------------
# AUTHOR: Isaiah Hessler
# FILENAME: decision_tree.py
# SPECIFICATION: Complete the given python program (decision_tree.py) that will read the file contact_lens.csv and output a decision tree.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 Hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
for row in db:
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

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
for row in db:
  if row[-1] == 'Yes':
    Y.append(1)
  else:
    Y.append(2)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()