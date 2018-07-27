import numpy as np
import math
import random


class KalmanFilter(object):
    '''
       Implements the following equations:
          X' = AX + BU + W  -> State Prediction   (' deontes estimate)
          P' = APA^ + Q     -> Covar Prediction   (^ denotes transpose)
          Y = Z - HX'       -> measurement diff
          S = HP'H^ + R     -> innovation covar
          K = PH^S"         -> kalman gain        (" denotes inverse)
          X = X'+ KY        -> update X
          P = (I - KH)P'    -> update P
          
       X: state matrix
       P: process covariance matrix
       W: noise in the state
       Q: noise in the process
       Y: risidual
       Z: measurement inputs
       H: observation matrix
       S: total system noise
       R: measurement noise
       K: kalman gain
       I: identity
    '''
       
    def __init__(self, X, P):
        pass

    
