import torch
import math
import numpy as np

class Gaussion:
    """Represents a gaussian distribution.
    """
    def __init__(self, mu, log_sigma=None) -> None:
        if log_sigma is None:
            mu, log_sigma = torch.chunk(mu, 2, -1)
        self.mu = mu
        if isinstance(log_sigma, torch.Tensor):
            self.log_sigma = torch.clamp(log_sigma, min=-10, max=2)
        else:
            self.log_sigma = np.clip(log_sigma, a_min=-10, a_max=2)
        self._sigma = None
    
    def sample(self):
        """Sample from this distribution.

        Returns:
            torch.Tensor: value.
        """
        return self.mu + self.sigma * torch.randn_like(self.sigma)
    
    def rsample(self):
        """Identical to self.sample(), to conform with pytorch naming scheme."""
        return self.sample()
    
    
    def kl_divergence(self, other):
        """Computes the KL divergence with other distribution.

        Args:
            other (Gaussion): other gaussion distribution.

        Returns:
            torch.Tensor: kl divergence.
        """
        return (other.log_sigma - self.log_sigma) + (self.sigma**2 + (self.mu - other.mu)**2) / (2 * other.sigma**2) - 0.5
    
    def log_prob(self, val):
        """Computes the log-probability of a value under the Gaussion distribution.

        Args:
            val (any): value
        """
        return -1 * ((val - self.mu) ** 2) / (2 * self.sigma**2) - self.log_sigma - math.log(math.sqrt(2*math.pi))
    
    def nll(self, x):
        """Negative log likehood (probability)

        Args:
            x (Any): value

        Returns:
            torch.Tensor: -log_prob
        """
        return -self.log_prob(x)
    
    def entropy(self):
        """Computes entropy of this distribution.

        Returns:
            torch.Tensor: entropy
        """
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.sigma)

    @property
    def sigma(self):
        if self._sigma is None:
            self._sigma = self.log_sigma.exp()
        return self._sigma

    @property
    def shape(self):
        return self.mu.shape
    
    @staticmethod
    def _combine(fcn, *argv, dim):
        mu, log_sigma = [], []
        for g in argv:
            mu.append(g.mu)
            log_sigma.append(g.log_sigma)
        mu = fcn(mu, dim)
        log_sigma = fcn(log_sigma, dim)
        return Gaussion(mu, log_sigma)
    
    @staticmethod
    def stack(*argv, dim):
        return Gaussion._combine(fcn=torch.stack, *argv, dim=dim)
    
    @staticmethod
    def cat(argv, dim):
        return Gaussion._combine(fcn=torch.cat, *argv, dim=dim)

    def tensor(self):
        return torch.cat([self.mu, self.log_sigma], dim=-1)
    
    def detach(self):
        """Detaches internal variables.

        Returns:
            Guassion: detached Gaussion.
        """
        return type(self)(self.mu.detach(), self.log_sigma.detach())
    

class MutivariateGuassion(Gaussion):
    def log_prob(self, val):
        return super().log_prob(val).sum(-1)
    
    @staticmethod
    def stack(*argv, dim):
        return MutivariateGuassion(Gaussion.stack(*argv, dim=dim).tensor())
    
    @staticmethod
    def cat(*argv, dim):
        return MutivariateGuassion(Gaussion.cat(*argv, dim=dim).tensor())