import numpy as np
import math


class KF(object):

    def __init__(self, groundTruth, initialEstimate, initialEstimateError, measurementError):
        self.trueVal = groundTruth
        self.estimate = initialEstimate
        self.estimateError = initialEstimateError
        self.measurement = 0.0
        self.measurementError = measurementError
        self.previousEstimate = initialEstimate
        self.previousEstimateError = initialEstimateError
        self.kg = 0.0
        return

    def SetPrevious(self):
        self.previousEstimate = self.estimate
        self.previousEstimateError = self.estimateError
        return

    def UpdateKalmanGain(self):
        self.kg = self.estimateError / (self.estimateError + self.measurementError)
        return

    def UpdateEstimate(self, measurement):
        self.measurement = measurement
        self.estimate = self.previousEstimate + self.kg*(self.measurement - self.previousEstimate)
        return

    def UpdateUncertainty(self):
        self.estimateError = (1 - self.kg)*self.previousEstimateError
        return



if(__name__ == "__main__"):
    mu = 72 #true value
    sigma = 2.1

    measurements = np.random.normal(mu, sigma, 200)

    kf = KF(mu, 120, 2, 4)

    for i in range(len(dist)):
        kf.UpdateKalmanGain()
        kf.UpdateEstimate(measurements[i])
        kf.UpdateUncertainty()
        kf.SetPrevious()
        print("{:.2f}    {:.2f}".format(kf.estimate, kf.estimateError))


















