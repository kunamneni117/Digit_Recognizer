import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

input_layer_size  = 784
hidden_layer_size = 25
num_labels = 10

data = pd.read_csv('train.csv')

X= data.iloc[:,1:].values
y = data.iloc[:,0].values

y = y.flatten()
 
m = X.shape[0]

def Random_Initialization(L_in,L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out,L_in + 1)* 2 *epsilon_init - epsilon_init
    return W


Theta1 = Random_Initialization(input_layer_size,hidden_layer_size)
Theta2 = Random_Initialization(hidden_layer_size,num_labels)

Theta_nn_params = np.hstack((Theta1.T.ravel(),Theta2.T.ravel()))


def SigmoidGradient(z):
    g = expit(z)*(1-expit(z))
    return g

def Cost(nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
    Theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels, (hidden_layer_size + 1))
    
    #m = np.size(X,0)
    m = y.size
    a1 = np.concatenate((np.ones((m,1)),X),axis = 1)
    z2 = a1.dot(Theta1.T)
    l2 = np.size(z2,0)
    
    a2 = np.concatenate((np.ones((l2,1)),expit(z2)),axis = 1)
    
    z3 = a2.dot(Theta2.T)
    a3 = expit(z3)
    
    
    yt = np.zeros((m,num_labels))
    
    yt[np.arange(m),y-1] =1
    
    cost = np.sum(-yt*np.log(a3) - (1-yt)*np.log(1-a3))
    
    delta3 = a3 -yt
    
    delta2 = delta3.dot(Theta2)* SigmoidGradient(np.concatenate((np.ones((l2,1)),z2),axis = 1))
    
    Theta2_grad = delta3.T.dot(a2)
    Theta1_grad = delta2[:,1:].T.dot(a1)
    
    cost = cost/m
    Theta2_grad = Theta2_grad/m
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + Lambda/m *Theta2[:,1:]
    Theta1_grad = Theta1_grad/m
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + Lambda/m *Theta1[:,1:]
    
    reg_cost = np.sum(np.power(Theta1[:,1:],2)) + np.sum(np.power(Theta2[:,1:],2))
    
    cost = cost + 1/(2*m)*Lambda* reg_cost
    
    grad = np.concatenate((Theta1_grad.flatten(),Theta2_grad.flatten()))
    
    return cost,grad

    
    
Lambda =1

costfunc = lambda p: Cost(p,input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[0]

gradfunc = lambda p: Cost(p,input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[1]


result = minimize(costfunc,Theta_nn_params, method ='CG', jac = gradfunc, options ={'disp' : True, 'maxiter' :50})

nn_params = result.x
cost = result.fun

Theta_1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],(hidden_layer_size, input_layer_size + 1),order = 'F').copy()
Theta_2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],(num_labels, (hidden_layer_size + 1)), order='F').copy()

test_data = pd.read_csv('test.csv')

#Xtest = test_data.iloc[:,1:].values
#ytest = test_data.iloc[:,0].values


def predict(Theta1,Theta2,Xtest):
    m = np.size(Xtest,0)
    Xtest = np.concatenate((np.ones((m,1)),Xtest), axis =1)
    
    Temp1 = expit(Xtest.dot(Theta1.T))
    temp_1 = np.concatenate((np.ones((m,1)),Temp1),axis =1)
    Temp2 = expit(temp_1.dot(Theta2.T))
    
    p = np.argmax(Temp2,axis =1) + 1
    
    return p

pred = predict(Theta_1,Theta_2,test_data)

#accuracy = np.mean(np.double(pred==y))*100

    

    