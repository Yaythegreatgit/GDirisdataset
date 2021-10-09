from sklearn.datasets import load_iris
import numpy as np
import math
from matplotlib import pyplot as plt

#############################################################################################
# Author: Ioannis Ypsilantis
# Date: 10/5/21
# Description: This is an application of logistic regression to evaluate the iris data-set
#              two classes, virginica and non-virginica are used, features sepal length and
#              sepal width are studied only. Determines best w, guaranteeing at least 65%
#              accuracy and plots data with decision boundary
# Pledge: I pledge my honor that I have abided by the Stevens Honor System.
#############################################################################################

#weight vector and learning rate, modify these to play with application, I found 0.1 to be the best rate
w = np.array([1,-1.0])
rate = 0.1

def gradient(): #Gradient of cross entropy
    counter = np.array([0.0,0.0])
    for i in range(len(train_data)): #essentially sumnation
        a = (sigmoid(train_data[i]) - train_target[i])
        counter += a * train_data[i]
    return counter

def sigmoid(n): #sigmoid function
    return 1/(1+math.exp(-np.dot(w, n)))

def accuracy(): #tests accuracy by comparing to test data returns number correct/total tests
    counter = 0
    for i in range(len(test_data)):
        if sigmoid(test_data[i]) > 0.5:
            if test_target[i] == 1:
                counter = counter + 1
        else:
            if test_target[i] == 0:
                counter = counter + 1
    return counter/len(test_data)

def graph(): #Graphs data with decision boundary
    mask1 = np.array([True, False]) #x (Sepal Length)
    mask2 = np.array([False, True]) #y (Sepal Width)
    x = np.linspace(4,8,100)
    y = -w[0]*x/w[1] #line of decision boundary define by this function

    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")

    plt.scatter(data[:, mask1], data[:,mask2], c=target, cmap='Spectral') #Spectral is convenient for two color selection using map
    plt.plot(x,y, label = "Decision Boundary")
    plt.legend()
    plt.show()
            


#dataset
iris = load_iris()
data = iris.data
target = iris.target

#Fix class definitions, versicolor should be 1 and non-versicolor should be 
target = np.where(target == 2, 1, 0) #2 is versicolor, 1 and 0 are non-versicolor so we relabel accordingly

#Mask isolate the two features we will be using
mask = np.array([True, True, False, False])
data = data[:, mask]

#Separate data and target into a training set and testing set
selection = np.random.randint(2, size = 150) #array randomly set to 0 - 1
train = np.where(selection == 1) #array of every index where selection was 1
test = np.where(selection == 0)  #array of every index where selection was 0

#with train and test we can select the data and targets at the indeces in train for training and test for testing.
#While selection is randomly generated every time, train_data[i] corresponds to train_target[i] and test_data[i] corresponds to test_target[i] always
train_data = data[train]
train_target = target[train]
test_data = data[test]
test_target = target[test]



#Defined gradient descent, need to compute gradient through function call
counter = 0
while counter <=5: #runs entil accuracy exceeds 65% five times in a row
    w = w - rate*gradient() #Stochastic gradient decsent
    if accuracy() > .65:
        counter += 1
    else:
        counter = 0
    rate *= .999 #slowly decrease the rate over time
print("Trained classifier: ")
print(w)
print("Accuracy: ")
print(accuracy())
graph()

#looking at the plot, it looks pretty muddy a lot of points overlap with eachother making it impossible for a highly accurate
#linear model. To this end, the model sometimes decides to always pick a single type because it can meet the 65% accuracy
#requirement.





