import matplotlib.pyplot as plt
import numpy as np
import math as mt

#Linear regression by Andres Escobedo


OrderOfFunc = 2 #Order of function to solve for

theta = np.zeros(OrderOfFunc) #Initialize theta with zeros

numIterations = 1000 #Number of iterations for gradient descent

alpha = 0.01 #Gradient descent step size


class LoadData():
    fil = open("LinRegData.txt", "r")

    if fil.mode == 'r':

        #Split the data into x and y array
        xDat = fil.readline().strip().split(" ")
        yDat = fil.readline().strip().split(" ")

        x = np.ones((len(xDat), 2))
        y = np.ones((len(xDat), 1))

        #convert to ints
        for i in range(0, len(x) - 1):
            x[i, 1] = float(xDat[i])

        for i in range(0, len(y) - 1):
            y[i, 0] = float(yDat[i])

        #print ("size of x = " + str(len(x)))
        #print ("size of y = " + str(len(y)))
        #print(x)
        #print(y)





#Display the initial dataset
def LoadDataToPlots():
    #Check that the data is of the same length
    if len(Data.x) == len(Data.y):

        plt.scatter(Data.x[:,1],Data.y)
        plt.xlabel('x')
        plt.ylabel('y')

        #Set axis numeration
        plt.xticks(np.arange(min(Data.x[:,1]), max(Data.x[:,1]) + 1, 1.0))
        plt.yticks(np.arange(min(Data.y), max(Data.y) + 1, 1.0))
        #plt.show()

    else:

        print("x and y are not the same size")

#Slow
def GetHypothesis():
    hyp = np.zeros((m,1))
    hypRow = np.matmul(theta, np.transpose(Data.x))
    for i in range(m):
        hyp[i,0] = hypRow[i]

    return hyp

#returns the value of the cost function
def GetCost():
    return np.sum(np.square(np.subtract(GetHypothesis(), Data.y)))/(2*m)

#Adjusts theta accordint to the gradient of the cost function
def GradientDescent():
    for j in range(len(theta)):
        theta[j] = theta[j] - alpha*np.sum(np.multiply(np.subtract(GetHypothesis(), Data.y),np.reshape(Data.x[:,j],(m,1))))/m


#Instantiate class which has all the data
Data = LoadData()

#Number of training sets
m = int(len(Data.y))

#Show the plots of data if it has only 1 feature
LoadDataToPlots()

#Apply gradient descent for the specified number of iterations
for i in range(numIterations):
    GradientDescent()

#Arrays to plot the result
GuessesX = np.zeros(mt.floor(max(Data.x[:,1])))
GuessesY = np.zeros(mt.floor(max(Data.x[:,1])))

for i in range(mt.floor(max(Data.x[:,1]))):
    GuessesX[i] = i
    GuessesY[i] = theta[0] + theta[1]*i

plt.plot(GuessesX, GuessesY, 'r')
plt.show()




