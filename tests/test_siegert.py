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
        """Generates random parameters for the siegert model
        """
        self.sigma_e = np.random.uniform(1,3)
        self.sigma_s = np.random.uniform(1,3)
        self.sigma_eta = np.random.uniform(1,3)

        self.beta = np.random.uniform(0.1,1)

        self.mu_y = np.random.uniform(3,5)
        self.mu_x = np.random.uniform(3,5)

        self.n_samples = 10000
        truth, ensemble = clim_snr.stat_model_generator.siegert(self.n_samples,np.random.randint(100,200,size=[self.n_samples]),mu_y = self.mu_y, mu_x = self.mu_x, beta = self.beta, sigma_e = self.sigma_e, sigma_s = self.sigma_s, sigma_eta = self.sigma_eta)
        
        self.truth = truth
        self.ensemble = ensemble



    def test_truth_variance(self):
        """Checks whether variance of the truth is within 3 standard deviations of its sampling uncertainty to the sum of the signal and error variance for the truth
        """

        estimated_variance = self.truth.var(ddof=0).values
        error_of_variance = estimated_variance*np.sqrt(2/(self.n_samples-1))
        
        np.testing.assert_allclose(self.sigma_e**2 + self.sigma_s**2, estimated_variance, atol=error_of_variance*3)


    def test_truth_mean(self):
        """Checks whether mean of the truth is within 3 standard deviations of its sampling uncertainty to the true mean (mu_y)        """

        estimated_variance = self.truth.var(ddof=0).values
        error_of_mean = estimated_variance/np.sqrt(self.n_samples)
        
        np.testing.assert_allclose(self.mu_y, self.truth.mean(dim=("sample")), atol=error_of_mean*3)



if __name__ == '__main__':
    unittest.main()