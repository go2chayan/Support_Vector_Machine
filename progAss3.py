# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:27:01 2014
Programming Assignment 3: Support Vector Machine with SGD
@author: Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
"""

import random
import numpy as np

def randomShuffle(X,Y):
    N = np.size(X,0)
    idx = range(N)
    np.random.shuffle(idx)
    return X[idx,0:], Y[idx]

def trainSVM(Xtr,Ytr,C,eta_0):
    timeStamp = 1.
    N = np.size(Xtr,0)
    d = np.size(Xtr,1)
    w = np.zeros([d,1])
    b = 0.
    eta = float(eta_0)/timeStamp
    converged = False
    # This loop will terminate after maximum iteration or when w converged
    # I tried changing the max timestamp from 3 to 300 but looks like
    # increasing the max timestamp doesn't have any positive impact on the
    # accuracy result
    while (timeStamp < 5) and not converged:
        # I tried random shuffling the data but looks like thats not a good
        # thing to do because in that case I am comparing the C and eta
        # on a variable basis. That means it is not legitimate to compare those
        # values anymore if I shuffle the dataset.
        #Xtr,Ytr = randomShuffle(Xtr,Ytr)    # Random shuffle the test data
        for n in range(N):
            Xn = Xtr[n,0:][None].T
            prevW = w
            # Applying stochastic gradiant descend update
            if (1 - Ytr[n]*(w.T.dot(Xn)) + b) > 0:
                w = w - eta*((1./N)*w - C*Ytr[n]*Xn)
                b = b + eta*(float(C)*Ytr[n])
            else:
                w = w - eta*(1./N)*w
            # debug
#            print 'Change in w =', (prevW - w).T.dot(prevW - w)/eta_0
#            print 'Eta =',eta
#            print 'Time Stamp =',timeStamp
            
            # w is assumed to be converged when it doesn't change much with
            # respect to eta_0
            if ((prevW - w).T.dot(prevW - w)/eta_0 < 1e-5):
                converged = True
                break
        # Updating time stamp and changing eta proportionally
        timeStamp = timeStamp + 1
        eta = eta_0/timeStamp
    return w,b
    
def testSVM(Xtest, Ytest, w,b):
    Yhat = np.sign(Xtest.dot(w) + b)
    acc = sum(Yhat == Ytest)/float(np.size(Xtest,0))
    return acc

def main():
    trainN = 348
    testN = 42
    devN = 45
    
    # Read each line and convert into 2D array
    dictionary = {'democrat':1,'republican':-1,'y':1,'n':-1,'?':0}
    input_file = open('voting2.dat')
    lines = input_file.readlines()
    allData = [line.strip().split(',') for line in lines if (\
        line.startswith('republican') or \
        line.startswith('democrat'))]
    for i in range(np.size(allData,0)):
        for j in range(np.size(allData,1)):
            allData[i][j] = dictionary[allData[i][j]]
    allData = np.array(allData)

    # Create training, development and test set by random sampling
    allIdx = set(range(np.size(allData,0)))
    trainingSet = random.sample(allIdx,trainN)
    devSet = random.sample(allIdx.difference(trainingSet),devN)
    testSet = random.sample(allIdx.difference(trainingSet).difference(devSet),testN)
    X_tr = allData[trainingSet,1:]
    Y_tr = allData[trainingSet,0][None].T
    X_dev = allData[devSet,1:]
    Y_dev = allData[devSet,0][None].T
    X_te = allData[testSet,1:]
    Y_te = allData[testSet,0][None].T
            
    # Building SVM model on training dataset and tuning C and eta on dev set
    # Searching C and eta in an exponential grid as suggested in
    # http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
    i=0
    j=0
    accuGrid = np.zeros([7,9])
    CGrid = np.zeros([7])
    etaGrid = np.zeros([9])
    
    # Tuning C and Learning rate here
    for C in np.power(2.,range(-7,7,2)):
        for Eta0 in np.arange(0.2,1.1,0.1):#np.power(2.,range(-5,5,2)):
            print 'Tuning ... ', 'C =',C, 'Eta_0 =',Eta0,
            w,b = trainSVM(X_tr,Y_tr,C,Eta0)  # Model building on training set
            accuGrid[i,j] = testSVM(X_dev,Y_dev,w,b)
            print 'Accuracy = ', accuGrid[i,j]
            CGrid[i] = C # storing C
            etaGrid[j] = Eta0; # storing learning rate
            j=j+1
        j=0
        i=i+1
    i_,j_ = np.where(accuGrid == np.max(accuGrid))
# this idea don't work
#    C_final = np.mean(CGrid[i_])    # Taking the mean of all the best C's
#    eta_final = np.mean(etaGrid[j_]) # Taking the mean of all the best eta's
    C_final = CGrid[i_[0]]
    eta_final = etaGrid[j_[0]]
    
    # Calculating the final model
    w_final, b_final = trainSVM(X_tr,Y_tr,C_final,eta_final)

    # Testing the final model in test data
    accuracy = testSVM(X_te,Y_te,w_final,b_final)
    print 'Accuracy in test data =',accuracy

if __name__ == '__main__':
    main()
    
