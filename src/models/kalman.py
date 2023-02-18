import numpy as np

"""
class KalmanFilter:
    def __init__(self, x0, dt):
        self.x = np.array([[x0], [0.]])
        self.u = np.array([[0.], [0.]])
        self.P = np.array([[0., 0.], [0., 100]])
        self.F = np.array([[1., dt], [0., 1]])
        self.H = np.array([[1., 0], [0, 0]])
        self.R = np.array([[0.1, 0], [0, 0]])
        self.I = np.identity((len(self.x)))

    def predict(self):
        self.x = np.dot(self.F, self.x) + self.u
        self.P = np.dot(np.dot(self.F, self.P), self.F.T)
        return self.x.reshape(2,)[0]

    def update(self, xt):
        Z = np.array([xt])
        y = Z.T - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.pinv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
        return self.x.reshape(2,)[0]
"""


class KalmanFilter:
    def __init__(self, x0, dt=1):
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([1, 0]).reshape(1, 2)
        self.Q = np.diag([1, 1])
        self.R = np.array([25]).reshape(1, 1)
        self.x = x0
        self.P = np.diag([1, 1])

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0]

    def update(self, z):
        K = np.dot(np.dot(self.P, self.H.T),
                   np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x[0]
