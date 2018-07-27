import pylab
import math
import random
import numpy as np

'''
kinematic equations:

x_pos(n) = x_pos(n-1) + x_dot(n-1)*dt + .5ax*dt^2
x_dot(n) = x_dot(n-1) + ax*dt - f*x_dot(n-1)         NOTE: f is drag constant
y_pos(n) = y_pos(n-1) + y_dot(n-1)*dt + .5ay*dt^2
y_dot(n) = y_dot(n-1) + ay*dt - f*y_dot(n-1)

since there is no x-acceleration, those terms will not be included in the matricies below

'''

#
true_x = []
true_y = []

pred_x = []
pred_y = []

meas_x = []
meas_y = []

dt = .1 #s
a = -9.81 #m/s^2
v0 = 100 #m/s initial velocity

x_pos = 0 #m initial x position
x_dot = v0*math.cos(math.pi/4) #m/s initial x velocity
y_pos = 0 #m initial y position
y_dot = v0*math.sin(math.pi/4) #m/s initial y velocity


#state transition matrix
A = np.matrix([[1, dt, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, dt],
               [0, 0, 0, 1]])

#state variables
X = np.matrix([[x_pos],
               [x_dot],
               [y_pos],
               [y_dot]])

#control matrix
B = np.matrix([[0],
               [0],
               [.5*dt**2],
               [dt]])

#control vector
U = np.matrix([a])


#measurement vector
Y = np.matrix([[0],
               [0],
               [0],
               [0]])

#measurement noise
W = np.matrix([[random.gauss(X[0,0], 6)],
               [random.gauss(X[0,0], 4)],
               [random.gauss(X[0,0], 6)],
               [random.gauss(X[0,0], 5)]])

#covariance matrix
P = np.matrix([[3, 0, 0, 0],
               [0, 2, 0, 0],
               [0, 0, 3, 0],
               [0, 0, 0, 2]])

#observation matrix
H = np.eye(4)

#process error covariance matrix
Q = np.eye(4)*.1

#measurement error covariance matrix
R = np.eye(4)*.1


print(X)
print(X[0])

X_hat = X

for i in range(0, 150):

    #simulate taking noisy measurements
    Y[0,0] = random.gauss(X[0,0], 6)
    Y[1,0] = random.gauss(X[1,0], 4)
    Y[2,0] = random.gauss(X[2,0], 6)
    Y[3,0] = random.gauss(X[3,0], 5)

    #add them to our plots
    meas_x.append(Y[0,0])
    meas_y.append(Y[2,0])
    true_x.append(X[0,0])
    true_y.append(X[2,0])
    
    X = A*X + B*U
 #   P = A*P*np.transpose(A) + Q
   # Y = H*Y
  #  y = Y - H*X
   # S = H*P*np.transpose(H) + R
  #  K = P*np.transpose(H)*np.linalg.inv(S)
  #  X = X + K*y
   # P = (np.eye(P.shape[0]) - K*H)*P

#print(true_x)

pylab.plot(true_x,true_y, '-')
pylab.plot(meas_x, meas_y, '.')
pylab.show()









