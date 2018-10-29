import numpy as np


class Noise(object):
    # ================================
    #    ORNSTEIN-UHLENBECK PROCESS
    # ================================
    def __init__(self, noise, dim):
        self.noise = noise
        self.dim = dim
        self.ou_theta = 0.15
        self.ou_mu = 0
        self.ou_sigma = 0.2

    def compute_ou_noise(self):
        # Solve using Euler-Maruyama method
        self.noise = self.noise + self.ou_theta * (self.ou_mu - self.noise) + self.ou_sigma * np.random.randn(self.dim)
        return self.noise

    def ou_noise(self):
        return self.noise
