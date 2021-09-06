import unittest
import os
import sys
import numpy as np
import clim_snr.stat
import clim_snr.stat_model_generator


class TestSiegert(unittest.TestCase):
    """Test Class for Siegert Data generation

    """
    @classmethod
    def setUpClass(self):
        self.sigma_e = np.random.uniform(1,3)
        self.sigma_s = np.random.uniform(1,3)
        self.sigma_eta = np.random.uniform(1,3)

        self.beta = np.random.uniform(0.1,1)

        self.mu_y = np.random.uniform(3,5)
        self.mu_x = np.random.uniform(3,5)

        n_samples = 10000
        truth, ensemble = clim_snr.stat_model_generator.siegert(n_samples,np.random.randint(100,200,size=[n_samples]),mu_y = self.mu_y, mu_x = self.mu_x, beta = self.beta, sigma_e = self.sigma_e, sigma_s = self.sigma_s, sigma_eta = self.sigma_eta)
        
        self.truth = truth
        self.ensemble = ensemble



    def test_truth_variance(self):
        """Checks whether variance of the truth is close to the sum of the signal and error variance for the truth
        """
        np.testing.assert_allclose(self.sigma_e**2 + self.sigma_s**2, self.truth.var(ddof=0).values, atol=1)


    def test_truth_mean(self):
        np.testing.assert_allclose(self.mu_y, self.truth.mean(dim=("sample")), atol=0.1)



if __name__ == '__main__':
    unittest.main()