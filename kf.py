import numpy as np
import math


class KF(object):

    def __init__(self):
        self.trueVal = 0.0
        self.estimate = 0.0
        self.estimateError = 0.0
        self.measurement = 0.0
        self.measurementError = 0.0
        self.previousEstimate = 0.0
        self.previousEstimateError = 0.0
        self.kg = 0.0
        return

    def Initialize(self):
        #self.trueVal = float(input("enter ground truth: "))
        #self.previousEstimate = float(input("enter initial estimate: "))
        #self.previousEstimateError = float(input("enter initial estimate error: "))
        #self.measurement = float(input("enter initial measurement: "))
        #self.measurementError = float(input("enter measurement error: "))

        self.trueVal = 72
        self.previousEstimate = 68
        self.previousEstimateError = 2
        self.estimateError = 2
        self.measurement = 75
        self.measurementError = 4
        
        return

    def SetPrevious(self):
        self.previousEstimate = self.estimate
        self.previousEstimateError = self.estimateError
        return

    def UpdateKalmanGain(self):
        self.kg = self.estimateError / (self.estimateError + self.measurementError)
        return

    def UpdateEstimate(self):
        print(self.estimate)
        print(self.previousEstimate)
        print(self.kg)
        print(self.measurement)
        self.estimate = self.previousEstimate + self.kg*(self.measurement - self.previousEstimate)
        return

    def UpdateUncertainty(self):
        self.estimateError = (1 - self.kg)*self.previousEstimateError
        return









kf = KF()

kf.Initialize()
kf.UpdateKalmanGain()
kf.UpdateEstimate()
kf.UpdateUncertainty()
kf.SetPrevious()
print(kf.estimate)







