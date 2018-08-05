import matplotlib.pyplot as plt
import math
import random
import numpy as np

'''
kinematic equations:

x_pos(n) = x_pos(n-1) + x_dot(n-1)*dt + .5ax*dt^2
x_dot(n) = x_dot(n-1) + ax*dt - f*x_dot(n-1)         NOTE: f is drag coefficient
y_pos(n) = y_pos(n-1) + y_dot(n-1)*dt + .5ay*dt^2
y_dot(n) = y_dot(n-1) + ay*dt - f*y_dot(n-1)

since there is no x-acceleration, those terms in x_pos & y_pos will not be included
in the matricies below

drag is pseudo included in the control matrix (as a hack)
'''

# used to plot ground truth and basis for simulated measurements
X_true = np.zeros((150, 4))
X_meas = np.zeros((150,4))
X_pred = np.zeros((150,4))

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
               [(1/a)*-.65*dt],
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
P  = np.matrix([[1, .7, 0, 0],
               [.7, 1, 0, 0],
               [0, 0, 1, .7],
               [0, 0, .7, 1]])

#observation matrix
H = np.eye(4)

#process error covariance matrix
#Q = lambda: np.eye(4)*random.gauss(5,.5)
Q = np.eye(4)

#measurement error covariance matrix
R = np.matrix([[36, 0, 0, 0],
               [0, 36, 0, 0],
               [0, 0, 49, 0],
               [0, 0, 0, 16]])

##R = np.matrix([[64, 48, 0, 0],
##               [48, 64, 0, 0],
##               [0, 0, 64, 64],
##               [0, 0, 64, 64]]) 

#               6   6   5   3
#R = np.matrix([[36, 18, 30, 18],
#               [18, 9, 15, 9],
#               [30, 15, 25, 15],
#               [18, 9, 15, 9]])



X_hat = X
P_hat = P

#get ground truth 
for i in range(0, 150):
    X = A*X + B*U
    X_true[i] = np.transpose(X)
    
# reset X, now these are the initial predicitons
X = np.matrix([[9],
               [v0*math.cos(math.pi/4)],
               [40],
               [v0*math.sin(math.pi/4)]])

print(X.transpose().shape)
print(X_pred.shape)

for i in range(0, 150):
    if(i == 0):
        m = input()
        m = m.split(',')

    #add predictions to the plot
    X_pred[i] = X.transpose()

    #update predictions
    X_hat = A*X + B*U
    P_hat = A*P*np.transpose(A) + Q
    if(i == 0):
        print("xhat: ")
        print(X_hat)
        print("phat: ")
        print(P_hat)
    


    #simulate taking noisy measurements
    Y[0,0] = random.gauss(X_true[i,0], 9)
    Y[1,0] = random.gauss(X_true[i,1], 3)
    Y[2,0] = random.gauss(X_true[i,2], 9)
    Y[3,0] = random.gauss(X_true[i,3], 6)
    if(i == 0):
        Y[0,0] = int(m[0])
        Y[1,0] = int(m[1])
        Y[2,0] = int(m[2])
        Y[3,0] = int(m[3])    
        print(Y)
    else:
            #simulate taking noisy measurements
        Y[0,0] = random.gauss(X_true[i,0], 9)
        Y[1,0] = random.gauss(X_true[i,1], 3)
        Y[2,0] = random.gauss(X_true[i,2], 9)
        Y[3,0] = random.gauss(X_true[i,3], 6)

    #add measurements to our plots
    X_meas[i] = Y.transpose()
    
    #find the residual
    y = Y - H*X_hat

    #calculate kalman gain
    S = H*P_hat*np.transpose(H) + R
    K = P_hat*np.transpose(H)*np.linalg.inv(S)
    k.append(K[0,0])

    #update state and covariance
    X = X_hat + K*y
    P = (np.eye(P.shape[0]) - K*H)*P_hat

    if(i == 0):
        print("y: ")
        print(y)
        print("S: ")
        print(S)
        print("S-1:")
        print(np.linalg.inv(S))
        print("K:")
        print(K)
        print("X:")
        print(X)
        print("P")
        print(P)

    
plt.figure(1)
plt.subplot(221)
plt.xlabel("x-pos")
plt.ylabel("y-pos")
plt.plot(X_true[:,0],X_true[:,2], '-', label="true pos")
plt.plot(X_meas[:,0],X_meas[:,2], ':', label="measured pos")
plt.plot(X_pred[:,0],X_pred[:,2], '--', label = "predicted pos")
plt.xlim(xmin=-10)
plt.ylim(ymin=-10)
plt.legend()

plt.subplot(222)
plt.xlabel("iterations")
plt.ylabel("Kalman Gain")
plt.plot(k, '-')


plt.subplot(223)
plt.xlabel("iterations")
plt.ylabel("y velocity")
plt.plot(X_true[:,3], '-', label="true")
plt.plot(X_meas[:,3], ':', label="measured")
plt.plot(X_pred[:,3], '--', label="predicted")
plt.legend()

plt.subplot(224)
plt.xlabel("iterations")
plt.ylabel("x velocity")
plt.plot(X_true[:,1], '-', label="true")
plt.plot(X_meas[:,1], ':', label="measured")
plt.plot(X_pred[:,1], '--', label="predicted")
plt.legend()



plt.subplots_adjust(hspace=.35)

plt.show()









