import matplotlib.pyplot as plt
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
drag is also not used at this time

'''

# used to plot ground truth and basis for simulated measurements
X_true = np.zeros((150, 4))

pred_x = []
pred_y = []

meas_x = []
meas_y = []

k = []
p = []

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


#covariance matrix
P = np.eye(4)*50

#observation matrix
H = np.eye(4)

#process error covariance matrix
Q = np.eye(4)*3

#measurement error covariance matrix
R = np.matrix([[36, 0, 0, 0],
               [0, 9, 0, 0],
               [0, 0, 49, 0],
               [0, 0, 0, 16]])


X_hat = X
P_hat = P

#get ground truth 
for i in range(0, 150):
    X = A*X + B*U
    X_true[i] = np.transpose(X)
    
# reset X, now these are the initial predicitons
X = np.matrix([[10],
               [v0*math.cos(math.pi/4)],
               [120],
               [v0*math.sin(math.pi/4)]])


for i in range(0, 150):
    #add predictions to the plot
    pred_x.append(X[0,0])
    pred_y.append(X[2,0])

    #update predictions
    X_hat = A*X + B*U
    P_hat = A*P*np.transpose(A) + Q


    #simulate taking noisy measurements
    Y[0,0] = random.gauss(X_true[i,0], 12)
    Y[1,0] = random.gauss(X_true[i,1], 7)
    Y[2,0] = random.gauss(X_true[i,2], 11)
    Y[3,0] = random.gauss(X_true[i,3], 13)

    #add measurements to our plots
    meas_x.append(Y[0,0])
    meas_y.append(Y[2,0])

    #find the residual
    y = Y - H*X_hat

    #calculate kalman gain
    S = H*P_hat*np.transpose(H) + R
    K = P_hat*np.transpose(H)*np.linalg.inv(S)
    k.append(K[0,0])

    #update state and covariance
    X = X_hat + K*y
    P = (np.eye(P.shape[0]) - K*H)*P_hat
    p.append(P[0,0])

    
plt.figure(1)
plt.subplot(311)
plt.xlabel("x-pos")
plt.ylabel("y-pos")
plt.plot(X_true[:,0],X_true[:,2], '-', label="true pos")
plt.plot(meas_x, meas_y, '.', label="measured pos")
plt.plot(pred_x, pred_y, '-', label = "predicted pos")
plt.xlim(xmin=-10)
plt.ylim(ymin=-10)
plt.legend()

plt.subplot(312)
plt.xlabel("iterations")
plt.ylabel("Kalman Gain")
plt.plot(k, '-')


plt.subplot(313)
plt.xlabel("iterations")
plt.ylabel("covariance")
plt.plot(p, '-')

plt.subplots_adjust(hspace=.35)

plt.show()









